from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import os
import json
from datetime import datetime


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        
        # Episode reward tracking for training progress
        self.all_episode_rewards = []
        self.episode_count = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            # Track episode rewards for training progress
            self.all_episode_rewards.append(episode_return)
            self.episode_count += 1
            
            # Save episode rewards periodically (every 10 episodes)
            if self.episode_count % 10 == 0:
                self.save_episode_rewards_to_file()

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def save_episode_rewards_to_file(self):
        """Save episode rewards data to ablation_results directory."""
        # Create ablation_results directory if it doesn't exist
        results_dir = "ablation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine config name based on args
        config_name = getattr(self.args, 'name', 'pymarl2_training')
        if hasattr(self.args, 'env_args') and 'map_name' in self.args.env_args:
            config_name += f"_{self.args.env_args['map_name']}"
        
        # Create data structure
        reward_data = {
            "config_name": config_name,
            "episode_rewards": self.all_episode_rewards,
            "total_episodes": self.episode_count,
            "alignment_weight": getattr(self.args, 'alignment_weight', 0.0),
            "llm_type": getattr(self.args, 'llm_type', 'none'),
            "algo_type": getattr(self.args, 'learner', 'unknown'),
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "average_reward": sum(self.all_episode_rewards) / len(self.all_episode_rewards) if self.all_episode_rewards else 0,
            "min_reward": min(self.all_episode_rewards) if self.all_episode_rewards else 0,
            "max_reward": max(self.all_episode_rewards) if self.all_episode_rewards else 0,
            "last_10_avg": sum(self.all_episode_rewards[-10:]) / min(10, len(self.all_episode_rewards)) if self.all_episode_rewards else 0
        }
        
        # Save to file
        file_path = f"{results_dir}/{config_name}_episode_rewards.json"
        with open(file_path, 'w') as f:
            json.dump(reward_data, f, indent=2)
        
        print(f"📊 Episode rewards saved to: {file_path} (Episodes: {self.episode_count})")
        return file_path

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
