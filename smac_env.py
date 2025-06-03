# smac_env.py
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from smac.env import StarCraft2Env
import numpy as np

class RLLibSMACEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = StarCraft2Env(map_name=env_config.get("map_name", "3m"))
        self.env.reset()
        self.agents = list(range(self.env.n_agents))
        self.observation_space_dict = {i: self.env.get_obs_space() for i in self.agents}
        self.action_space_dict = {i: self.env.get_action_space() for i in self.agents}

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        return {i: obs[i] for i in self.agents}

    def step(self, action_dict):
        actions = [action_dict[i] for i in self.agents]
        reward, done, info = self.env.step(actions)
        obs = self.env.get_obs()
        rewards = {i: reward for i in self.agents}
        obs_dict = {i: obs[i] for i in self.agents}
        done_dict = {i: done for i in self.agents}
        done_dict["__all__"] = done
        return obs_dict, rewards, done_dict, info

    def observation_space(self, agent_id):
        return self.observation_space_dict[agent_id]

    def action_space(self, agent_id):
        return self.action_space_dict[agent_id]