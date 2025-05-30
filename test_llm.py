# test_llm.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from smac.env import StarCraft2Env
import sys
sys.path.append('./pymarl2/src/llm')
from translate import get_state_NL

import os
import openai
import random
import re
import traceback
import time
from llm_utils import setup_logging, format_llm_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Set up logging
logger = setup_logging()

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def __init__(self, model_name="llama3:latest", verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        openai.api_base = 'http://127.0.0.1:11434/v1'
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
        """Implement LLM API call with retries and robust parsing."""
        # Optionally print prompt
        if self.verbose:
            print("\n===== LLM PROMPT =====")
            print(prompt_text)
            print("======================")
        # Retry up to 3 times
        llm_output_str = ''
        for attempt in range(3):
            try:
                resp = openai.ChatCompletion.create(model=self.model_name,
                                                    messages=[{"role": "user", "content": prompt_text}])
                llm_output_str = resp.choices[0].message.content.strip()
                break
            except Exception as err:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {err}")
                time.sleep(1)
        # Display formatted response
        if self.verbose:
            out = format_llm_output(llm_output_str)
            print("\n===== LLM RESPONSE =====")
            print(out)
            print("========================")
        # Extract list patterns
        matches = re.findall(r"\[\s*(\d+(?:\s*,\s*\d+)*)\]", llm_output_str)
        if matches:
            best = max(matches, key=len)
            try:
                return eval(f"[{best}]")
            except:
                logger.error(f"Invalid action string: {best}")
        # Fallback random
        logger.warning("No valid action list—falling back to random")
        return [random.choice(all_available_action_indices[i]) if all_available_action_indices[i].size else 0
                for i in range(n_agents)]

    def act(self, global_state_nl, avail_actions_list, n_agents, env_info):
        n_total_actions = env_info["n_actions"]
        
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

class AgentNet(nn.Module):
    """Individual agent Q-network."""
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(AgentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class MixingNet(nn.Module):
    """QMIX mixing network (hypernetwork)."""
    def __init__(self, n_agents, state_dim, mixing_embed_dim):
        super(MixingNet, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetworks for generating weights and biases
        self.hyper_w1 = nn.Linear(state_dim, mixing_embed_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        
        # Generate mixing network weights and biases
        w1 = torch.abs(self.hyper_w1(states))  # Ensure positive weights
        b1 = self.hyper_b1(states)
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        # Reshape weights
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        
        # Forward pass through mixing network
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        y = torch.bmm(hidden, w2) + b2.view(batch_size, 1, 1)
        
        return y.view(batch_size, 1)

class ReplayBuffer:
    """Experience replay buffer for QMIX training."""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, obs, actions, reward, next_state, next_obs, done, avail_actions, next_avail_actions):
        # Store raw numpy arrays/values in the buffer
        experience = (state, obs, actions, reward, next_state, next_obs, done, avail_actions, next_avail_actions)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        # Sample a batch of experiences
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Unpack the batch
        states, obs, actions, rewards, next_states, next_obs, dones, avail_actions, next_avail_actions = zip(*batch)
        
        # Convert to PyTorch tensors (device placement will be done by caller)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(next_obs)),
            torch.BoolTensor(np.array(dones)),
            torch.FloatTensor(np.array(avail_actions)),
            torch.FloatTensor(np.array(next_avail_actions))
        )
        
    def __len__(self):
        return len(self.buffer)

class QMIXAgent:
    def __init__(self, n_agents=3, state_dim=None, obs_dim=None, n_actions=None, verbose=False, lr=0.0005, gamma=0.99, tau=0.005):
        """Initialize QMIX agent with neural networks."""
        self.verbose = verbose
        self.n_agents = n_agents
        self.n_actions = n_actions or 20  # Default value, will be updated
        self.state_dim = state_dim or 120  # Default value, will be updated  
        self.obs_dim = obs_dim or 40  # Default value, will be updated
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"QMIXAgent will use device: {self.device}")
        
        # Will be initialized when we get environment info
        self.agent_nets = None
        self.target_agent_nets = None
        self.mixing_net = None
        self.target_mixing_net = None
        self.optimizer = None
        self.replay_buffer = ReplayBuffer(max_size=10000)
        
        # Training parameters
        self.batch_size = 32
        self.train_freq = 5
        self.target_update_freq = 100
        self.step_count = 0
        
        if self.verbose:
            print(f"QMIXAgent initialized (networks not yet built)")
    
    def initialize_networks(self, env_info):
        """Initialize networks once we have environment information."""
        self.n_actions = env_info["n_actions"]
        self.state_dim = env_info["state_shape"]
        self.obs_dim = env_info["obs_shape"]
        
        hidden_dim = 64
        mixing_embed_dim = 32
        
        # Initialize agent networks and move to device
        self.agent_nets = nn.ModuleList([
            AgentNet(self.obs_dim, hidden_dim, self.n_actions).to(self.device) 
            for _ in range(self.n_agents)
        ])
        
        # Initialize target agent networks and move to device
        self.target_agent_nets = nn.ModuleList([
            AgentNet(self.obs_dim, hidden_dim, self.n_actions).to(self.device) 
            for _ in range(self.n_agents)
        ])
        
        # Initialize mixing networks and move to device
        self.mixing_net = MixingNet(self.n_agents, self.state_dim, mixing_embed_dim).to(self.device)
        self.target_mixing_net = MixingNet(self.n_agents, self.state_dim, mixing_embed_dim).to(self.device)
        
        # Copy weights to target networks
        for target_param, param in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            target_param.data.copy_(param.data)
            
        # Initialize optimizer
        all_params = list(self.agent_nets.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.lr)
        
        if self.verbose:
            print(f"QMIX networks initialized on {self.device}: state_dim={self.state_dim}, obs_dim={self.obs_dim}, n_actions={self.n_actions}")
    
    def act(self, global_state_nl, avail_actions_list, n_agents, env_info, obs=None):
        """Choose actions using QMIX network policy."""
        # Initialize networks if not done yet
        if self.agent_nets is None:
            self.initialize_networks(env_info)
        
        # If we don't have individual observations, fall back to random policy
        if obs is None:
            if self.verbose:
                print("QMIXAgent: No individual observations provided, using random policy")
            chosen_actions = []
            for i in range(n_agents):
                avail_agent_actions_mask = avail_actions_list[i]
                available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
                if len(available_action_indices) > 0:
                    chosen_actions.append(random.choice(available_action_indices))
                else:
                    chosen_actions.append(0)  # no-op if no actions available
            return chosen_actions
        
        chosen_actions = []
        
        with torch.no_grad():
            for i in range(n_agents):
                # Get individual agent observation and move to device
                agent_obs = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                
                # Get Q-values from agent network
                q_values = self.agent_nets[i](agent_obs)
                
                # Mask unavailable actions
                avail_actions_mask = torch.BoolTensor(avail_actions_list[i]).unsqueeze(0).to(self.device)
                q_values_masked = q_values.clone()
                q_values_masked[~avail_actions_mask] = -float('inf')  # Use ~ for boolean mask
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    # Random action from available actions
                    available_action_indices = np.nonzero(avail_actions_list[i])[0]
                    if len(available_action_indices) > 0:
                        action = random.choice(available_action_indices)
                    else:
                        action = 0
                else:
                    # Greedy action
                    action = q_values_masked.argmax().item()
                
                chosen_actions.append(action)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if self.verbose:
            print(f"QMIXAgent chosen actions: {chosen_actions} (epsilon: {self.epsilon:.3f})")
        
        return chosen_actions
    
    def train(self, state, obs, action, total_reward, next_state, next_obs, done, avail_actions, next_avail_actions):
        """Train QMIX networks with the given experience."""
        if self.agent_nets is None:
            if self.verbose:
                print("QMIXAgent: Networks not initialized, skipping training")
            return
            
        # Store experience in replay buffer
        self.replay_buffer.add(state, obs, action, total_reward, next_state, next_obs, done, avail_actions, next_avail_actions)
        
        self.step_count += 1
        
        # Train every train_freq steps
        if self.step_count % self.train_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._update_networks()
        
        # Update target networks
        if self.step_count % self.target_update_freq == 0:
            self._update_target_networks()
        
        if self.verbose and self.step_count % 100 == 0:  # Reduce frequency of logging
            print(f"QMIXAgent: Training step {self.step_count}, reward: {total_reward:.2f}, buffer size: {len(self.replay_buffer)}")
    
    def _update_networks(self):
        """Update QMIX networks using experience replay."""
        states, obs, actions, rewards, next_states, next_obs, dones, avail_actions, next_avail_actions = self.replay_buffer.sample(self.batch_size)
        
        # Move all tensors to device
        states = states.to(self.device)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)  # Shape: (batch_size, 1)
        next_states = next_states.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device).unsqueeze(-1)  # Shape: (batch_size, 1)
        avail_actions = avail_actions.to(self.device)
        next_avail_actions = next_avail_actions.to(self.device)
        
        batch_size = states.size(0)
        
        # Get current Q-values for each agent
        current_q_values = []
        for i in range(self.n_agents):
            agent_obs = obs[:, i, :]  # Shape: (batch_size, obs_dim)
            q_vals = self.agent_nets[i](agent_obs)  # Shape: (batch_size, n_actions)
            agent_actions = actions[:, i].unsqueeze(1)  # Shape: (batch_size, 1)
            q_val = q_vals.gather(1, agent_actions).squeeze(1)  # Shape: (batch_size,)
            current_q_values.append(q_val)
        
        current_q_values = torch.stack(current_q_values, dim=1)  # Shape: (batch_size, n_agents)
        
        # Get target Q-values for each agent
        target_q_values = []
        with torch.no_grad():
            for i in range(self.n_agents):
                next_agent_obs = next_obs[:, i, :]
                next_q_vals = self.target_agent_nets[i](next_agent_obs)
                
                # Mask unavailable actions using boolean mask
                next_avail = next_avail_actions[:, i, :].bool()  # Ensure it's boolean for masking
                next_q_vals_masked = next_q_vals.clone()
                next_q_vals_masked[~next_avail] = -float('inf')
                
                max_next_q = next_q_vals_masked.max(dim=1)[0]
                target_q_values.append(max_next_q)
        
        target_q_values = torch.stack(target_q_values, dim=1)  # Shape: (batch_size, n_agents)
        
        # Get mixed Q-values
        current_mixed_q = self.mixing_net(current_q_values, states)  # Shape: (batch_size, 1)
        target_mixed_q = self.target_mixing_net(target_q_values, next_states)  # Shape: (batch_size, 1)
        
        # Calculate TD target: r + gamma * Q_target * (1-done)
        target_q = rewards + self.gamma * target_mixed_q * (~dones)
        
        # Calculate loss
        loss = F.mse_loss(current_mixed_q, target_q)
        
        # Check for NaN loss
        if torch.isnan(loss).any():
            if self.verbose:
                print("WARNING: NaN loss detected, skipping update")
            return
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_nets.parameters()) + list(self.mixing_net.parameters()), max_norm=1.0)
        self.optimizer.step()
        
        if self.verbose and self.step_count % 50 == 0:
            print(f"QMIXAgent: Loss = {loss.item():.4f}")
    
    def _update_target_networks(self):
        """Soft update target networks."""
        # Update target agent networks
        for target_param, param in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Update target mixing network
        for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        if self.verbose:
            print("QMIXAgent: Target networks updated")

def calculate_alignment_reward(qmix_actions, llm_recommended_actions, n_agents):
    """Calculate alignment reward between QMIX actions and LLM recommended actions.
    
    Args:
        qmix_actions: List of action IDs chosen by QMIX
        llm_recommended_actions: List of action IDs recommended by LLM
        n_agents: Number of agents
    
    Returns:
        Float between 0.0 and 1.0 representing alignment ratio
    """
    if len(qmix_actions) != n_agents or len(llm_recommended_actions) != n_agents:
        return 0.0
    
    matches = sum(1 for qmix_act, llm_act in zip(qmix_actions, llm_recommended_actions) 
                  if qmix_act == llm_act)
    alignment_ratio = matches / n_agents
    
    return alignment_ratio

def run_smac_with_agents(llm_agent, qmix_agent, map_name, num_episodes, max_steps_per_episode, verbose_env=False, render=False, save_replay=False, alignment_weight=0.1):
    """Runs both LLM and QMIX agents collaboratively on the specified SMAC map.
    
    Args:
        llm_agent: LLMAgent instance for recommendations
        qmix_agent: QMIXAgent instance for action selection and training
        map_name: StarCraft II map name
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        verbose_env: Whether to print detailed environment information
        render: Whether to render the environment
        save_replay: Whether to save game replays
        alignment_weight: Weight for alignment reward in QMIX training
    """
    try:
        env = StarCraft2Env(map_name=map_name, 
                            replay_dir="replays" if save_replay else None,
                            debug=verbose_env)
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]
        
        # Explicitly initialize QMIX networks here
        qmix_agent.initialize_networks(env_info)

        print(f"Starting SMAC with LLM-QMIX collaboration on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode: {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print(f"Alignment reward weight: {alignment_weight}")
        print("-" * 50)

        total_rewards = []
        total_alignment_rewards = []
        
        for e_idx in range(num_episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            episode_alignment_reward = 0
            
            if verbose_env: 
                print(f"\n--- Episode {e_idx + 1} ---")
            
            for step in range(max_steps_per_episode):
                if render: 
                    env.render()

                # Get environment information
                global_state = env.get_state()
                obs = env.get_obs()  # Get individual agent observations
                avail_actions_list = env.get_avail_actions()
                global_state_nl = get_state_NL(env, global_state)

                # Get LLM recommendations
                llm_recommended_actions = llm_agent.act(global_state_nl, avail_actions_list, n_agents, env_info)
                
                # Get QMIX actions (pass individual observations)
                qmix_actions = qmix_agent.act(global_state_nl, avail_actions_list, n_agents, env_info, obs=obs)
                
                # Calculate alignment reward
                alignment_reward = calculate_alignment_reward(qmix_actions, llm_recommended_actions, n_agents)
                episode_alignment_reward += alignment_reward
                
                if verbose_env:
                    print(f"  Step {step + 1}:")
                    print(f"    LLM recommended: {llm_recommended_actions}")
                    print(f"    QMIX chosen: {qmix_actions}")
                    print(f"    Alignment: {alignment_reward:.2f}")

                # Execute QMIX actions in environment
                env_reward, terminated, info = env.step(qmix_actions)
                episode_reward += env_reward
                
                # Get next state and observations for training
                next_global_state = env.get_state() if not terminated else np.zeros_like(global_state)
                next_obs = env.get_obs() if not terminated else [np.zeros_like(o) for o in obs]
                next_avail_actions = env.get_avail_actions() if not terminated else [np.zeros_like(a) for a in avail_actions_list]
                
                # Calculate combined reward (environment + alignment)
                combined_reward = env_reward + alignment_weight * alignment_reward
                
                # Train QMIX agent directly with current experience
                qmix_agent.train(
                    state=global_state,
                    obs=obs,
                    action=qmix_actions,
                    total_reward=combined_reward,
                    next_state=next_global_state,
                    next_obs=next_obs,
                    done=terminated,
                    avail_actions=avail_actions_list,
                    next_avail_actions=next_avail_actions
                )
                
                if verbose_env:
                    print(f"    Env reward: {env_reward:.2f}, Total reward: {combined_reward:.2f}")
                    print(f"    Terminated: {terminated}")

                if terminated:
                    if save_replay:
                        env.save_replay()
                        if verbose_env: 
                            print(f"Replay saved for episode {e_idx + 1}.")
                    break
            
            # Episode summary
            win_status = "UNKNOWN"
            if 'battle_won' in info:
                win_status = "WON" if info['battle_won'] else "LOST/DRAW"
            
            avg_alignment = episode_alignment_reward / (step + 1) if step >= 0 else 0
            
            print("\n" + "=" * 50)
            print(f"Episode {e_idx + 1} Summary:")
            print("-" * 30)
            print(f"  Steps: {step + 1}")
            print(f"  Environment reward: {episode_reward:.2f}")
            print(f"  Average alignment: {avg_alignment:.2f}")
            print(f"  Status: {win_status}")
            print("=" * 50 + "\n")
            
            total_rewards.append(episode_reward)
            total_alignment_rewards.append(avg_alignment)

        print("-" * 50)
        print("LLM-QMIX collaboration test finished.")
        if total_rewards:
            print(f"Average environment reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
            print(f"  Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f}")
        if total_alignment_rewards:
            print(f"Average alignment reward: {np.mean(total_alignment_rewards):.2f}")
            print(f"  Min: {np.min(total_alignment_rewards):.2f}, Max: {np.max(total_alignment_rewards):.2f}")
        
        # Print a win rate if applicable
        win_count = sum(1 for r in total_rewards if r > 0)
        if num_episodes > 0:
            print(f"Win rate: {win_count/num_episodes:.2%} ({win_count}/{num_episodes})")
        
        print("-" * 50)

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2, SMAC, and dependencies are correctly installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")

def run_smac_with_agent(agent, map_name, episodes, max_steps_per_episode, render=False, verbose_env=False, save_replay=False):
    """Legacy function for single agent testing."""
    try:
        env = StarCraft2Env(map_name=map_name, 
                            replay_dir="replays" if save_replay else None,
                            debug=verbose_env)
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]

        print(f"Starting SMAC with {agent.__class__.__name__} on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode: {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print("-" * 30)

        total_rewards = []
        for e_idx in range(episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            
            if verbose_env: 
                print(f"\n--- Episode {e_idx + 1} ---")

            for step in range(max_steps_per_episode):
                if render: 
                    env.render()

                global_state = env.get_state()
                avail_actions_list = env.get_avail_actions()
                global_state_nl = get_state_NL(env, global_state)

                actions = agent.act(global_state_nl, avail_actions_list, n_agents, env_info)
                print("Chosen actions:", actions)
                reward, terminated, info = env.step(actions)
                episode_reward += reward

                if verbose_env:
                    print(f"  Step {step + 1}: Actions={actions}, Reward={reward:.2f}, Terminated={terminated}")

                if terminated:
                    if save_replay:
                        env.save_replay()
                        if verbose_env: 
                            print(f"Replay saved for episode {e_idx + 1}.")
                    break
            
            win_status = "UNKNOWN"
            if 'battle_won' in info:
                win_status = "WON" if info['battle_won'] else "LOST/DRAW"
            
            print(f"Episode {e_idx + 1} finished. Steps: {step + 1}. Reward: {episode_reward:.2f}. Status: {win_status}")
            total_rewards.append(episode_reward)

        print("-" * 30)
        print(f"{agent.__class__.__name__} test finished.")
        if total_rewards:
            print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f} (Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f})")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2, SMAC, and dependencies are correctly installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    MAP_NAME = "3m"  # Popular SMAC map
    NUM_EPISODES = 2
    MAX_STEPS = 100
    RENDER_ENV = False
    VERBOSE_AGENT = True  # Controls agent internal prints
    VERBOSE_ENV_LOOP = True  # Controls step-by-step prints in the main run loop
    ALIGNMENT_WEIGHT = 0.1  # Weight for alignment reward in QMIX training

    # Initialize both agents
    llm_agent = LLMAgent(verbose=VERBOSE_AGENT)
    qmix_agent = QMIXAgent(verbose=VERBOSE_AGENT)

    print("Running LLM-QMIX collaborative training...")
    
    # Run collaborative training
    run_smac_with_agents(
        llm_agent=llm_agent,
        qmix_agent=qmix_agent,
        map_name=MAP_NAME,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        render=RENDER_ENV,
        verbose_env=VERBOSE_ENV_LOOP,
        save_replay=True,
        alignment_weight=ALIGNMENT_WEIGHT
    )
