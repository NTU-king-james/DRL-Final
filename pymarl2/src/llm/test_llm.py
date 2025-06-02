from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from smac.env import StarCraft2Env
from .translate import get_state_NL

import os
import openai
import random

def get_action_description(action_id, n_total_actions):
    """Provides a basic textual description for an action ID."""
    if action_id == 0: return "no_op"
    if action_id == 1: return "stop"
    if n_total_actions > 2 and action_id == 2: return "move_north"
    if n_total_actions > 3 and action_id == 3: return "move_south"
    if n_total_actions > 4 and action_id == 4: return "move_east"
    if n_total_actions > 5 and action_id == 5: return "move_west"
    if n_total_actions > 6 and (6 <= action_id < n_total_actions):
        return f"attack_target_slot_{action_id - 6}"
    return f"action_id_{action_id}"

class LLMAgent:
    def __init__(self, model_name="remijang/smac-sft-gemma3", llm_type="ours", verbose=False):
        self.model_name = model_name
        self.llm_type = llm_type
        self.verbose = verbose
        openai.api_base = 'http://127.0.0.1:11434/v1' # test with ollama
        openai.api_key = 'ollama'


    def _construct_prompt(self, global_state_nl, avail_actions_list, n_agents, n_total_actions):
        prompt = (
            f"You are an AI agent controlling {n_agents} units in a StarCraft II scenario.\n"
            "Your objective is to defeat all enemy units by issuing commands to your units.\n"
            "--- CURRENT GLOBAL STATE ---\n"
            f"{global_state_nl}\n"
            "--- ORDERS FOR YOUR UNITS ---\n"
            f"For each of your {n_agents} units, select one action from its list of available actions.\n"
            "In the last line respond with a formatted list of integer action IDs, one for each unit.\n"
            "For example, to order agent 0, 1, 2 to perform action ID 0, 1, 5 respectively, output: [0, 1, 5]\n"
            "Make sure the action ID given to each agent is available for each agent."
        )

        all_agent_available_action_indices = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            all_agent_available_action_indices.append(available_action_indices)

            prompt += f"\n-- Unit {i} --\nAvailable Actions (ID: Description):\n"
            if len(available_action_indices) == 0:
                prompt += " - No actions available (unit may be incapacitated or have no valid moves).\n"
            else:
                for action_id in available_action_indices:
                    action_desc = get_action_description(action_id, n_total_actions)
                    prompt += f" - ID {action_id}: {action_desc}\n"
        
        prompt += f"\nBased on the state and available actions, provide the list of {n_agents} action IDs for your units."
        return prompt, all_agent_available_action_indices
    def _get_actions_from_llm_api(self, prompt_text, n_agents, all_available_action_indices, n_total_actions):
        """
        Implement your actual LLM API call here.
        This method should:
        1. Use `self.client` to send `prompt_text` to the LLM.
        2. Parse the LLM's response into a list of action integers.
        3. Validate that each action is in the `all_available_action_indices` for the respective agent.
        4. Handle errors, retries, and ensure the output is a list of `n_agents` integers.        Return a list of chosen action IDs.
        """
        if self.verbose:
            print("\n===== LLM PROMPT (Sending to API) =====")
            print(prompt_text)
            print("=======================================")
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                      {"role": "user", "content": prompt_text}
                  ]
            )
            llm_output_str = response.choices[0].message.content.strip()
            import re
            # Look for pattern like [1, 2, 3] or [7] in the response
            action_match = re.search(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]', llm_output_str)
            if action_match:
                action_str = "[" + action_match.group(1) + "]"
                if self.verbose:
                    print(f"LLM response:\n{llm_output_str}")
                    print(f"Parsed action string: {action_str}")
                return eval(action_str)
            else:
                # If no valid format found, try to extract numbers from the last line
                last_line = llm_output_str.split('\n')[-1]
                numbers = re.findall(r'\d+', last_line)
                if numbers:
                    result = [int(num) for num in numbers]
                    if self.verbose:
                        print(f"LLM response:\n{llm_output_str}")
                        print(f"Fallback parsing, extracted numbers: {result}")
                    return result
                else:
                    if self.verbose:
                        print(f"LLM response:\n{llm_output_str}")
                        print("No valid action format found")
                    return []
        except Exception as e:
            llm_output_str = locals().get('llm_output_str', None)
            if llm_output_str is not None:
                print("Parse action error!\nLLM output:", llm_output_str)
            else:
                print("Parse action error! No LLM output. Exception:", e)
            return []
    
    
    def act(self, global_state_nl, avail_actions_list, n_agents, env_info):
        n_total_actions = env_info["n_actions"]
        
        # Get available actions for each agent
        all_agent_available_action_indices = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            all_agent_available_action_indices.append(available_action_indices)
        
        # Check if this is random mode
        if self.llm_type == "random" or self.model_name == "random":
            # Random LLM: directly pick random actions from available actions
            chosen_actions = [random.choice(all_agent_available_action_indices[i]) for i in range(n_agents)]
            if self.verbose:
                action_descs_final = [get_action_description(ac, n_total_actions) for ac in chosen_actions]
                print(f"RandomLLMAgent selected actions: IDs={chosen_actions}, Descriptions={action_descs_final}")
            return chosen_actions
        
        # Normal LLM mode: construct prompt and call LLM API
        prompt, all_agent_available_action_indices = self._construct_prompt(
            global_state_nl, avail_actions_list, n_agents, n_total_actions
        )
        chosen_actions = list(self._get_actions_from_llm_api(prompt, n_agents, all_agent_available_action_indices, n_total_actions))

        # Basic validation for action list length
        if len(chosen_actions) != n_agents: # revert to random
            chosen_actions = [random.choice(all_agent_available_action_indices[i]) for i in range(n_agents)]
            if self.verbose:
                print(f"LLMAgent Warning: Action list length mismatch")
        for i in range(n_agents): # illegal moves, revert to random
            if chosen_actions[i] not in all_agent_available_action_indices[i]:
                
                chosen_actions[i] = random.choice(all_agent_available_action_indices[i])

        if self.verbose:
            action_descs_final = [get_action_description(ac, n_total_actions) for ac in chosen_actions]
            print(f"LLMAgent final selected actions: IDs={chosen_actions}, Descriptions={action_descs_final}")
        
        return chosen_actions

def run_smac_with_agent(agent, map_name="3m", episodes=1, max_steps_per_episode=200, render=False, verbose_env=False, save_replay=False):
    """Runs the given agent on the specified SMAC map."""
    try:
        env = StarCraft2Env(map_name=map_name, 
                            replay_dir="replays" if save_replay else None, # SMAC default replay dir is SC2Replays
                            debug=verbose_env)
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]

        print(f"Starting SMAC with {agent.__class__.__name__} on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode (set): {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print("-" * 30)

        total_rewards = []
        for e_idx in range(episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            
            if verbose_env: print(f"\n--- Episode {e_idx + 1} ---")

            for step in range(max_steps_per_episode):
                if render: env.render()

                # Get information for the agent
                # obs_list = env.get_obs() # Raw observations, if needed by agent
                global_state = env.get_state()
                avail_actions_list = env.get_avail_actions()
                
                global_state_nl = get_state_NL(env, global_state)

                # Agent chooses actions
                actions = agent.act(global_state_nl, avail_actions_list, n_agents, env_info)
                print("Chosen actions:", actions)
                reward, terminated, info = env.step(actions)
                episode_reward += reward

                if verbose_env:
                    print(f"  Step {step + 1}: Actions={actions}, Reward={reward:.2f}, Terminated={terminated}")

                if terminated:
                    if save_replay:
                        env.save_replay()
                        if verbose_env: print(f"Replay saved for episode {e_idx + 1}.")
                    break
            
            win_status = "UNKNOWN"
            if 'battle_won' in info: # info might be empty if episode ended due to step limit
                win_status = "WON" if info['battle_won'] else "LOST/DRAW"
            
            print(f"Episode {e_idx + 1} finished. Steps: {step + 1}. Reward: {episode_reward:.2f}. Status: {win_status}")
            total_rewards.append(episode_reward)

        print("-" * 30)
        print(f"{agent.__class__.__name__} test finished.")
        if total_rewards:
            print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f} (Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f})")
        else:
            print("No episodes were run or no rewards collected.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2, SMAC, and dependencies (like 'translate.py') are correctly installed/accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    MAP_NAME = "3m"  # Popular SMAC map
    NUM_EPISODES = 1
    MAX_STEPS = 100
    RENDER_ENV = False
    VERBOSE_AGENT = False  # Controls LLMAgent's internal prints (prompts, chosen actions)
    VERBOSE_ENV_LOOP = False # Controls step-by-step prints in the main run loop

    # Initialize the LLM Agent
    llm_agent = LLMAgent(verbose=VERBOSE_AGENT)

    run_smac_with_agent(
        agent=llm_agent,
        map_name=MAP_NAME,
        episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        render=RENDER_ENV,
        verbose_env=VERBOSE_ENV_LOOP,
        save_replay=True 
    )