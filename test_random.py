from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from smac.env import StarCraft2Env
from translate import get_obs_NL, get_state_NL
# To use the observation describer, you would uncomment the next line
# and ensure the smac_obs_describer.py file (or the describe_smac_observation function)
# is accessible in your Python path.
# from smac_obs_describer import describe_smac_observation, get_example_smac_observation



        
def run_random_agent_smac(map_name="2s_vs_1sc", episodes=1, max_steps_per_episode=200, render=False, verbose=False,save_replay=False):
    """
    Runs a random agent on the specified SMAC map.

    Args:
        map_name (str): The name of the SMAC map to run (e.g., "8m", "3s5z").
        episodes (int): The number of episodes to run.
        max_steps_per_episode (int): The maximum number of steps for each episode.
        render (bool): Whether to render the environment (if supported by your SC2 setup).
                       Note: Rendering often requires specific setup and might not work out-of-the-box.
        verbose (bool): If True, prints more detailed information during execution,
                        including textual observation descriptions if the describer is enabled.
    """
    try:
        env = StarCraft2Env(map_name=map_name,replay_dir="replays")
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_actions = env_info["n_actions"] # Total number of possible discrete actions for any agent

        print(f"Starting SMAC environment with map: {map_name}")
        print(f"Number of agents: {n_agents}")
        print(f"Number of possible actions per agent: {n_actions}")
        print(f"Episode limit (max steps from env): {env_info.get('episode_limit', 'N/A')}")
        print("-" * 30)

        total_rewards = []

        for e in range(episodes):
            env.reset()
            print(env.agents[0])
            terminated = False
            episode_reward = 0
            episode_steps = 0

            if verbose:
                print(f"\n--- Episode {e + 1} ---")
                
            for step in range(max_steps_per_episode):
                print('a')
                if render:
                    env.render() # May require X server or specific setup
                print('b')
                # Get observations and available actions for each agent
                obs_list = env.get_obs() # List of individual agent observations
                global_state = env.get_state() # Global state
                enemy_count = env.get_obs_enemy_feats_size()[0]
                ally_count = env.get_obs_ally_feats_size()[0] + 1
                nf_al = env.get_ally_num_attributes()
                nf_en = env.get_enemy_num_attributes()
                print("Global State:\n")
                print(get_state_NL(env,global_state))
                obs_nl_list = get_obs_NL(env,obs_list)
                avail_actions_list = env.get_avail_actions()
                for i, obs in enumerate(obs_nl_list):
                    print(f"Agent {i} obs:\n")
                    print(obs)
                actions = []
                for agent_id in range(n_agents):
                    avail_agent_actions = avail_actions_list[agent_id]
                    # Get indices of available actions (where mask is 1)
                    available_action_indices = np.nonzero(avail_agent_actions)[0]

                    if len(available_action_indices) == 0:
                        chosen_action = 0 # Default to action 0 (often 'stop' or 'no_op')
                    else:
                        # Choose a random action from the available ones
                        chosen_action = np.random.choice(available_action_indices)
                    
                    actions.append(chosen_action)
                
                # Execute actions
                reward, terminated, info = env.step(actions)
                episode_reward += reward
                episode_steps += 1

                if verbose:
                    print(f"  Step {step + 1}: Actions={actions}, Reward={reward:.2f}, Terminated={terminated}")


                if terminated:
                    if save_replay:
                        env.save_replay() # save replay for each episode
                    break
            
            if render: # Keep window open for a bit after episode ends if rendering
                import time
                time.sleep(1)

            total_rewards.append(episode_reward)
            print(f"Episode {e + 1} finished after {episode_steps} steps. Reward: {episode_reward:.2f}")

        print("-" * 30)
        print("Random agent test finished.")
        if total_rewards:
            print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
            print(f"Min reward: {np.min(total_rewards):.2f}, Max reward: {np.max(total_rewards):.2f}")
        else:
            print("No episodes were run or no rewards collected.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2 and SMAC are correctly installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    # --- Configuration ---
    MAP_NAME = "3s5z"  # Example map, try "3m", "5m_vs_6m", "MMM" etc.
                     # Ensure the map is downloaded/available in your SC2 Maps directory.
    NUM_EPISODES = 1
    MAX_STEPS = 1
    RENDER_ENV = False # Set to True to attempt rendering (requires SC2 GUI & correct setup)
    VERBOSE_LOGGING = True # Set to True for step-by-step details

    # For rendering to work, StarCraft II needs to be installed, and PySC2 needs to be
    # configured to find it. Rendering also typically requires a graphical environment (X server on Linux).
    
    run_random_agent_smac(
        map_name=MAP_NAME,
        episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        render=RENDER_ENV,
        verbose=VERBOSE_LOGGING
    )