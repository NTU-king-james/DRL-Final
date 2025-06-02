# test_llm.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import sys
import os
# Use local SMAC environment instead of pymarl2
from smac.env import StarCraft2Env
from translate_1 import get_state_NL

import openai
import random
import re
import traceback
import time
import argparse
from llm_utils import setup_logging, format_llm_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import json
from tqdm import tqdm
from prompt import *

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

def get_action_id(action_desc):
    """Convert action description back to action ID."""
    action_desc = action_desc.lower().strip()
    
    # Handle common action mappings
    action_mappings = {
        "no_op": 0,
        "noop": 0,
        "nothing": 0,
        "stop": 1,
        "stay": 1,
        "stand": 1,
        "move_north": 2,
        "north": 2,
        "move_south": 3,
        "south": 3,
        "move_east": 4,
        "east": 4,
        "move_west": 5,
        "west": 5
    }
    
    # Check direct mappings first
    if action_desc in action_mappings:
        return action_mappings[action_desc]
    
    # Handle attack actions
    if "attack" in action_desc:
        # Look for patterns like "attack_1", "attack_target_slot_0", etc.
        if action_desc.startswith("attack_target_slot_"):
            try:
                slot_num = int(action_desc.split("_")[-1])
                return 6 + slot_num
            except:
                return 0  # Fallback to no_op
        elif action_desc.startswith("attack_"):
            try:
                target_num = int(action_desc.split("_")[-1])
                return 6 + target_num - 1  # Convert 1-based to 0-based for slot
            except:
                return 0  # Fallback to no_op
        else:
            # Generic attack - try to extract number
            import re
            numbers = re.findall(r'\d+', action_desc)
            if numbers:
                try:
                    target_num = int(numbers[0])
                    return 6 + target_num - 1  # Convert 1-based to 0-based for slot
                except:
                    return 0
    
    # Handle action_id_ format
    if action_desc.startswith("action_id_"):
        try:
            action_id = int(action_desc.split("_")[-1])
            return action_id
        except:
            return 0  # Fallback to no_op
    
    return 0  # Default fallback to no_op

class LLMAgent:
    def __init__(self, model_name="llama3:latest", llm_type="llama3", verbose=False):
        self.model_name = model_name
        self.llm_type = llm_type
        self.verbose = verbose
        
        if llm_type != "none":
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
        # Always show when LLM is being queried for action tracing
        print(f"\n🧠 QUERYING LLM ({self.model_name}) for {n_agents} agents...")
        
        # Optionally print full prompt in verbose mode
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
                print(f"❌ LLM API call failed (attempt {attempt+1}): {err}")
                time.sleep(1)
        # Always display LLM response for action tracing
        print("\n🤖 LLM RESPONSE:")
        print("=" * 60)
        out = format_llm_output(llm_output_str)
        print(out)
        print("=" * 60)
        
        # Extract list patterns
        matches = re.findall(r"\[\s*(\d+(?:\s*,\s*\d+)*)\]", llm_output_str)
        if matches:
            best = max(matches, key=len)
            try:
                parsed_actions = eval(f"[{best}]")
                print(f"🎯 LLM PARSED ACTIONS: {parsed_actions}")
                return parsed_actions
            except:
                print(f"❌ PARSE ERROR: Invalid action string: {best}")
        
        # Fallback random
        print("⚠️  LLM FALLBACK: No valid action list found, using random actions")
        fallback_actions = [random.choice(all_available_action_indices[i]) if all_available_action_indices[i].size else 0
                           for i in range(n_agents)]
        print(f"🎲 RANDOM FALLBACK ACTIONS: {fallback_actions}")
        return fallback_actions

    def act(self, global_state_nl, avail_actions_list, n_agents, env_info):
        n_total_actions = env_info["n_actions"]
        
        # Handle different LLM types
        if self.llm_type == "random":
            return self._random_actions(avail_actions_list, n_agents)
        elif self.llm_type == "none":
            # Return no-op actions when LLM is disabled
            return [0] * n_agents
        
        prompt, all_agent_available_action_indices = self._construct_prompt(
            global_state_nl, avail_actions_list, n_agents, n_total_actions
        )
        chosen_actions = list(self._get_actions_from_llm_api(prompt, n_agents, all_agent_available_action_indices, n_total_actions))

        # Basic validation for action list length
        if len(chosen_actions) != n_agents: # revert to random
            print(f"⚠️  ACTION COUNT MISMATCH: Expected {n_agents}, got {len(chosen_actions)}")
            chosen_actions = [random.choice(all_agent_available_action_indices[i]) for i in range(n_agents)]
            print(f"🎲 FALLBACK RANDOM ACTIONS: {chosen_actions}")
        
        for i in range(n_agents): # illegal moves, revert to random
            if chosen_actions[i] not in all_agent_available_action_indices[i]:
                old_action = chosen_actions[i]
                chosen_actions[i] = random.choice(all_agent_available_action_indices[i])
                print(f"❌ INVALID ACTION: Agent {i} action {old_action} → {chosen_actions[i]} (random)")

        # Always show final selected actions prominently for progress tracking
        action_descs_final = [get_action_description(ac, n_total_actions) for ac in chosen_actions]
        print("\n✅ LLM FINAL ACTIONS:")
        print(f"   🎮 IDs: {chosen_actions}")
        print(f"   📝 Actions: {action_descs_final}")
        
        return chosen_actions
    
    def _random_actions(self, avail_actions_list, n_agents):
        """Generate random but valid actions for each agent."""
        chosen_actions = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            if len(available_action_indices) > 0:
                chosen_actions.append(random.choice(available_action_indices))
            else:
                chosen_actions.append(0)  # no-op if no actions available
        
        if self.verbose:
            print(f"🎲 Random agent actions: {chosen_actions}")
        
        return chosen_actions

class RandomAgent:
    """Random agent for baseline comparison."""
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def act(self, global_state_nl, avail_actions_list, n_agents, env_info, obs=None):
        """Choose random valid actions for each agent."""
        chosen_actions = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            if len(available_action_indices) > 0:
                chosen_actions.append(random.choice(available_action_indices))
            else:
                chosen_actions.append(0)  # no-op if no actions available
        
        if self.verbose:
            print(f"RandomAgent chosen actions: {chosen_actions}")
        
        return chosen_actions

class AgentNet(nn.Module):
    """Individual agent Q-network with deeper architecture for better GPU utilization."""
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(AgentNet, self).__init__()
        # Deeper network for more GPU compute
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Additional layer
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)  # Bottleneck layer
        self.fc5 = nn.Linear(hidden_dim // 2, n_actions)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_values = self.fc5(x)
        return q_values

class MixingNet(nn.Module):
    """QMIX mixing network (hypernetwork) with enhanced architecture for better GPU utilization."""
    def __init__(self, n_agents, state_dim, mixing_embed_dim):
        super(MixingNet, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Enhanced hypernetworks with more layers for better GPU utilization
        hypernet_hidden = mixing_embed_dim * 2  # Larger intermediate layer
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, mixing_embed_dim * n_agents)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, mixing_embed_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, mixing_embed_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, 1)
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
        self.epsilon_decay = 0.9  # Per-episode decay (was 0.995 per step)
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"QMIXAgent will use device: {self.device}")
            if torch.cuda.is_available():
                # Warm up GPU and show initial stats
                warm_up_gpu()
                gpu_stats = check_gpu_utilization()
                if gpu_stats:
                    print(f"GPU Memory: {gpu_stats['allocated_gb']:.2f}GB allocated / {gpu_stats['total_gb']:.2f}GB total")
        
        # Will be initialized when we get environment info
        self.agent_nets = None
        self.target_agent_nets = None
        self.mixing_net = None
        self.target_mixing_net = None
        self.optimizer = None
        self.replay_buffer = ReplayBuffer(max_size=50000)  # Increased buffer size
        
        # Training parameters - Optimized for better GPU utilization
        self.batch_size = 128  # Increased from 32 to better utilize GPU
        self.train_freq = 1    # Train every step instead of every 5 steps
        self.target_update_freq = 100
        self.step_count = 0
        
        if self.verbose:
            print(f"QMIXAgent initialized (networks not yet built)")
    
    def initialize_networks(self, env_info):
        """Initialize networks once we have environment information."""
        self.n_actions = env_info["n_actions"]
        self.state_dim = env_info["state_shape"]
        self.obs_dim = env_info["obs_shape"]
        
        # Larger network dimensions for better GPU utilization
        hidden_dim = 128  # Increased from 64
        mixing_embed_dim = 64  # Increased from 32
        
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
        """Choose actions using proper QMIX centralized action selection."""
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
        
        # Epsilon-greedy at joint action level
        if random.random() < self.epsilon:
            # Random joint action from available actions
            chosen_actions = []
            for i in range(n_agents):
                available_action_indices = np.nonzero(avail_actions_list[i])[0]
                if len(available_action_indices) > 0:
                    chosen_actions.append(random.choice(available_action_indices))
                else:
                    chosen_actions.append(0)
        else:
            # QMIX coordinated action selection
            chosen_actions = self._qmix_action_selection(obs, avail_actions_list, global_state_nl, env_info)
        
        # Epsilon decay moved to episode-level (removed from step-level)
        
        if self.verbose:
            print(f"QMIXAgent chosen actions: {chosen_actions} (epsilon: {self.epsilon:.3f})")
        
        return chosen_actions
    
    def _qmix_action_selection(self, obs, avail_actions_list, global_state_nl, env_info):
        """Centralized action selection using QMIX mixing network."""
        with torch.no_grad():
            # Get global state tensor
            global_state = torch.FloatTensor(env_info.get('state', np.zeros(self.state_dim))).unsqueeze(0).to(self.device)
            
            # Generate all valid joint actions
            valid_joint_actions = self._generate_valid_joint_actions(avail_actions_list)
            
            # If too many joint actions, use greedy individual selection as fallback
            if len(valid_joint_actions) > 1000:  # Limit computational complexity
                return self._greedy_individual_selection(obs, avail_actions_list)
            
            best_joint_action = None
            best_q_tot = -float('inf')
            
            # Evaluate each valid joint action
            for joint_action in valid_joint_actions:
                # Get Q-values for each agent given this joint action
                agent_q_values = []
                for i in range(self.n_agents):
                    agent_obs = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                    q_vals = self.agent_nets[i](agent_obs)
                    agent_q_val = q_vals[0, joint_action[i]]  # Q-value for this agent's action
                    agent_q_values.append(agent_q_val)
                
                # Stack agent Q-values and get total Q-value from mixing network
                agent_q_tensor = torch.stack(agent_q_values).unsqueeze(0)  # Shape: (1, n_agents)
                q_tot = self.mixing_net(agent_q_tensor, global_state)
                
                # Track best joint action
                if q_tot.item() > best_q_tot:
                    best_q_tot = q_tot.item()
                    best_joint_action = joint_action
            
            if self.verbose:
                print(f"QMIXAgent: Evaluated {len(valid_joint_actions)} joint actions, best Q_tot: {best_q_tot:.3f}")
            
            return best_joint_action if best_joint_action is not None else self._greedy_individual_selection(obs, avail_actions_list)
    
    def _generate_valid_joint_actions(self, avail_actions_list):
        """Generate all valid joint actions given availability constraints."""
        from itertools import product
        
        # Get available actions for each agent
        agent_available_actions = []
        for i in range(self.n_agents):
            available_indices = np.nonzero(avail_actions_list[i])[0]
            if len(available_indices) == 0:
                available_indices = [0]  # Fallback to no-op
            agent_available_actions.append(available_indices.tolist())
        
        # Generate all combinations (Cartesian product)
        joint_actions = list(product(*agent_available_actions))
        return joint_actions
    
    def _greedy_individual_selection(self, obs, avail_actions_list):
        """Fallback to individual greedy action selection when joint action space is too large."""
        chosen_actions = []
        
        for i in range(self.n_agents):
            # Get individual agent observation and move to device
            agent_obs = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            
            # Get Q-values from agent network
            q_values = self.agent_nets[i](agent_obs)
            
            # Mask unavailable actions
            avail_actions_mask = torch.BoolTensor(avail_actions_list[i]).unsqueeze(0).to(self.device)
            q_values_masked = q_values.clone()
            q_values_masked[~avail_actions_mask] = -float('inf')
            
            # Greedy action selection
            action = q_values_masked.argmax().item()
            chosen_actions.append(action)
        
        if self.verbose:
            print("QMIXAgent: Used individual greedy selection (joint action space too large)")
        
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
        
        # Enhanced monitoring with GPU stats
        if self.verbose and self.step_count % 50 == 0:
            gpu_stats = check_gpu_utilization()
            print(f"QMIXAgent: Loss = {loss.item():.4f}")
            if gpu_stats:
                print(f"GPU: {gpu_stats['allocated_gb']:.2f}GB/{gpu_stats['total_gb']:.2f}GB ({gpu_stats['utilization_percent']:.1f}%)")
        
        # Force GPU synchronization to ensure proper utilization measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
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
    
    def save_weights(self, episode_num, config_name):
        """Save model weights for a specific episode and configuration."""
        if self.agent_nets is None:
            if self.verbose:
                print("QMIXAgent: Cannot save weights - networks not initialized")
            return None
        
        import os
        
        # Create ablation_results main directory if it doesn't exist
        os.makedirs("ablation_results", exist_ok=True)
        
        # Create weights directory path based on ablation study config
        weights_dir = f"ablation_results/{config_name}/weights"
        os.makedirs(weights_dir, exist_ok=True)
        
        # Define save paths (only save network weights, not optimizer)
        agent_weights_path = f"{weights_dir}/agent_nets_episode_{episode_num}.pth"
        mixing_weights_path = f"{weights_dir}/mixing_net_episode_{episode_num}.pth"
        
        # Save agent networks
        agent_state_dicts = [net.state_dict() for net in self.agent_nets]
        torch.save(agent_state_dicts, agent_weights_path)
        
        # Save mixing network
        torch.save(self.mixing_net.state_dict(), mixing_weights_path)
        
        if self.verbose:
            print(f"QMIXAgent: Weights saved for episode {episode_num} in {weights_dir}")
        
        return weights_dir
    
    def load_weights(self, episode_num, config_name):
        """Load model weights from a specific episode and configuration."""
        import os
        
        # Define load paths
        weights_dir = f"ablation_results/{config_name}/weights"
        agent_weights_path = f"{weights_dir}/agent_nets_episode_{episode_num}.pth"
        mixing_weights_path = f"{weights_dir}/mixing_net_episode_{episode_num}.pth"
        optimizer_path = f"{weights_dir}/optimizer_episode_{episode_num}.pth"
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [agent_weights_path, mixing_weights_path, optimizer_path]):
            print(f"QMIXAgent: Warning - Some weight files not found for episode {episode_num} in {weights_dir}")
            return False
        
        # Ensure networks are initialized
        if self.agent_nets is None:
            print("QMIXAgent: Cannot load weights - networks not initialized. Call initialize_networks() first.")
            return False
        
        try:
            # Load agent networks
            agent_state_dicts = torch.load(agent_weights_path, map_location=self.device)
            for net, state_dict in zip(self.agent_nets, agent_state_dicts):
                net.load_state_dict(state_dict)
            
            # Load mixing network
            self.mixing_net.load_state_dict(torch.load(mixing_weights_path, map_location=self.device))
            
            # Load optimizer state
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            
            # Copy to target networks
            for target_param, param in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_param.data.copy_(param.data)
            
            if self.verbose:
                print(f"QMIXAgent: Successfully loaded weights from episode {episode_num}")
            return True
            
        except Exception as e:
            print(f"QMIXAgent: Error loading weights from episode {episode_num}: {e}")
            return False

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

def run_smac_with_agents(llm_agent, qmix_agent, map_name, num_episodes, max_steps_per_episode, verbose_env=False, render=False, save_replay=False, alignment_weight=0.1, mode='train', load_episode=None, config_name=None):
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
        mode: 'train' (save weights every 10 episodes) or 'test' (load weights)
        load_episode: Episode number to load weights from (test mode only)
        config_name: Configuration name for weight saving/loading
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
        
        # Handle weight loading for test mode
        if mode == 'test' and load_episode is not None and config_name is not None:
            print(f"Loading weights from episode {load_episode} for config {config_name}...")
            if qmix_agent.load_weights(load_episode, config_name):
                print("✅ Weights loaded successfully")
            else:
                print("❌ Failed to load weights - continuing with random initialization")
        elif mode == 'test':
            print("⚠️  Test mode specified but missing load_episode or config_name - using random initialization")

        print(f"Starting SMAC with LLM-QMIX collaboration on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode: {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print(f"Alignment reward weight: {alignment_weight}")
        print(f"Mode: {mode.upper()}" + (f" (loading from episode {load_episode})" if mode == 'test' and load_episode else ""))
        print("-" * 50)

        # Determine config name based on LLM type and algorithm (if not provided)
        if config_name is None:
            if llm_agent is None or llm_agent.llm_type == 'none':
                config_name = "baseline_pure_qmix"
                llm_type = "none"
            else:
                llm_type = llm_agent.llm_type
                if alignment_weight == 0.0:
                    if llm_type == "llama3":
                        config_name = "baseline_pure_llm_pretrained"
                    elif llm_type == "ours":
                        config_name = "baseline_pure_llm_finetuned"
                    elif llm_type == "random":
                        config_name = "baseline_pure_llm_random"
                    else:
                        config_name = f"baseline_pure_llm_{llm_type}"
                else:
                    if llm_type == "llama3":
                        config_name = "qmix_pretrained_llm"
                    elif llm_type == "ours":
                        config_name = "qmix_finetuned_llm"
                    elif llm_type == "random":
                        config_name = "qmix_random_llm"
                    else:
                        config_name = f"qmix_{llm_type}_llm"
        else:
            # Use provided config_name and determine llm_type
            llm_type = llm_agent.llm_type if llm_agent is not None else "none"

        total_rewards = []
        total_alignment_rewards = []
        battle_wins = []  # Track actual battle outcomes for consistent win rate calculation
        
        # Create episode progress bar
        episode_pbar = tqdm(range(num_episodes), desc=f"🏆 Episodes ({map_name})", unit="ep", 
                           ncols=100, colour="green")
        
        for e_idx in episode_pbar:
            env.reset()
            terminated = False
            episode_reward = 0
            episode_alignment_reward = 0
            
            if verbose_env: 
                print(f"\n--- Episode {e_idx + 1} ---")
            
            # Create step progress bar for current episode
            step_pbar = tqdm(range(max_steps_per_episode), desc=f"⚔️  Ep{e_idx+1} Steps", 
                            unit="step", ncols=80, colour="blue", leave=False)
            
            for step in step_pbar:
                if render: 
                    env.render()

                # Get environment information
                global_state = env.get_state()
                obs = env.get_obs()  # Get individual agent observations
                avail_actions_list = env.get_avail_actions()
                global_state_nl = get_state_NL(env, global_state)

                # Get QMIX actions (pass individual observations)
                qmix_actions = qmix_agent.act(global_state_nl, avail_actions_list, n_agents, env_info, obs=obs)
                
                # Determine LLM interaction frequency based on LLM type
                llm_interaction_condition = False
                if alignment_weight > 0.0:
                    # All LLMs (including fine-tuned 'ours') interact every 5 steps
                    llm_interaction_condition = (step % 5 == 0)
                
                if llm_interaction_condition:
                    # Get LLM recommendations
                    llm_recommended_actions = llm_agent.act(global_state_nl, avail_actions_list, n_agents, env_info)
                    
                    # Calculate alignment reward only when LLM provides recommendations
                    alignment_reward = calculate_alignment_reward(qmix_actions, llm_recommended_actions, n_agents)
                    episode_alignment_reward += alignment_reward
                else:
                    # No LLM interaction this step: no alignment reward
                    llm_recommended_actions = None
                    alignment_reward = 0.0
                
                if verbose_env:
                    print(f"  Step {step + 1}:")
                    if alignment_weight > 0.0 and llm_interaction_condition:
                        print(f"    LLM interaction step (every 5 steps for {llm_agent.llm_type})")
                        print(f"    LLM recommended: {llm_recommended_actions}")
                        print(f"    QMIX chosen: {qmix_actions}")
                        print(f"    Alignment: {alignment_reward:.2f}")
                    elif alignment_weight > 0.0:
                        print(f"    No LLM interaction this step")
                        print(f"    QMIX chosen: {qmix_actions}")
                        print(f"    Alignment: 0.0 (no LLM guidance)")
                    else:
                        print(f"    QMIX chosen: {qmix_actions}")
                        print(f"    Pure QMIX mode: No LLM involvement")

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
                    step_pbar.close()  # Close step progress bar when episode terminates
                    break
            
            # Close step progress bar if we reached max steps without termination
            if not terminated:
                step_pbar.close()
            
            # Episode summary
            win_status = "UNKNOWN"
            battle_won = False  # Default to False
            if 'battle_won' in info:
                battle_won = info['battle_won']
                if battle_won:
                    win_status = "WON"
                else:
                    # Check if it's a draw (no enemies and no allies) or a loss
                    if hasattr(env, '_episode_count') and env._episode_count >= env.episode_limit:
                        win_status = "DRAW"  # Episode limit reached without decisive victory
                    else:
                        win_status = "LOST"
            
            # Calculate average alignment based on steps with LLM interaction
            total_steps = step + 1
            if alignment_weight > 0.0:
                # All LLMs interact every 5 steps (0, 5, 10, 15, ...)
                llm_interaction_steps = (total_steps + 4) // 5
            else:
                llm_interaction_steps = 0
            avg_alignment = episode_alignment_reward / llm_interaction_steps if llm_interaction_steps > 0 else 0
            
            print("\n" + "=" * 50)
            print(f"Episode {e_idx + 1} Summary:")
            print("-" * 30)
            print(f"  Steps: {step + 1}")
            print(f"  Environment reward: {episode_reward:.2f}")
            if alignment_weight > 0.0:
                print(f"  LLM interaction steps: {llm_interaction_steps}/{step + 1}")
                print(f"  Average alignment: {avg_alignment:.2f}")
            else:
                print(f"  Pure QMIX mode: No alignment tracking")
            print(f"  Status: {win_status}")
            print("=" * 50 + "\n")
            
            total_rewards.append(episode_reward)
            total_alignment_rewards.append(avg_alignment)
            battle_wins.append(battle_won)  # Track actual battle outcome
            
            # Decay epsilon once per episode
            if qmix_agent.epsilon > qmix_agent.epsilon_min:
                qmix_agent.epsilon *= qmix_agent.epsilon_decay
                if verbose_env:
                    print(f"Epsilon decayed to: {qmix_agent.epsilon:.4f}")

            # Save weights every 2 episodes in training mode
            if mode == 'train' and config_name is not None and qmix_agent is not None and (e_idx + 1) % 3 == 0:
                print(f"💾 Saving weights for episode {e_idx + 1}...")
                saved_dir = qmix_agent.save_weights(e_idx + 1, config_name)
                if saved_dir:
                    print(f"✅ Weights saved successfully for episode {e_idx + 1} in {saved_dir}")
                else:
                    print(f"❌ Failed to save weights for episode {e_idx + 1}")
            elif mode == 'train' and config_name is None:
                print(f"⚠️  Warning: config_name is None, cannot save weights for episode {e_idx + 1}")
            elif mode == 'train' and qmix_agent is None:
                print(f"⚠️  Warning: qmix_agent is None, cannot save weights for episode {e_idx + 1}")

        # Close the episode progress bar
        episode_pbar.close()

        print("-" * 50)
        if alignment_weight > 0.0:
            print("LLM-QMIX collaboration test finished.")
        else:
            print("Pure QMIX test finished.")
        
        if total_rewards:
            print(f"Average environment reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
            print(f"  Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f}")
        
        if alignment_weight > 0.0 and total_alignment_rewards:
            print(f"Average alignment reward: {np.mean(total_alignment_rewards):.2f}")
            print(f"  Min: {np.min(total_alignment_rewards):.2f}, Max: {np.max(total_alignment_rewards):.2f}")
        elif alignment_weight == 0.0:
            print("No alignment rewards tracked (Pure QMIX mode)")
        
        # Print a win rate if applicable
        actual_win_count = sum(battle_wins)  # Count actual battle victories
        if num_episodes > 0:
            print(f"Win rate: {actual_win_count/num_episodes:.2%} ({actual_win_count}/{num_episodes})")
        
        print("-" * 50)
        
        # Save episode rewards data for analysis
        # Determine config name based on LLM type and algorithm
        if llm_agent is None or llm_agent.llm_type == 'none':
            config_name = "baseline_pure_qmix"
            llm_type = "none"
        else:
            llm_type = llm_agent.llm_type
            if alignment_weight == 0.0:
                if llm_type == "llama3":
                    config_name = "baseline_pure_llm_pretrained"
                elif llm_type == "ours":
                    config_name = "baseline_pure_llm_finetuned"
                elif llm_type == "random":
                    config_name = "baseline_pure_llm_random"
                else:
                    config_name = f"baseline_pure_llm_{llm_type}"
            else:
                if llm_type == "llama3":
                    config_name = "qmix_pretrained_llm"
                elif llm_type == "ours":
                    config_name = "qmix_finetuned_llm"
                elif llm_type == "random":
                    config_name = "qmix_random_llm"
                else:
                    config_name = f"qmix_{llm_type}_llm"
        
        # Save the episode rewards
        algo_type = "qmix" if qmix_agent is not None else "none"
        save_episode_rewards(config_name, total_rewards, num_episodes, alignment_weight, llm_type, algo_type, battle_wins)

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
                            debug=verbose_env,
                            )  # Legacy function doesn't use LLM
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]

        print(f"Starting SMAC with {agent.__class__.__name__} on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode: {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print("-" * 30)

        total_rewards = []
        battle_wins = []  # Track actual battle outcomes for consistent win rate calculation
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
            
            # Episode summary
            win_status = "UNKNOWN"
            battle_won = False  # Default to False
            if 'battle_won' in info:
                battle_won = info['battle_won']
                if battle_won:
                    win_status = "WON"
                else:
                    # Check if it's a draw (episode limit reached) or a loss
                    if step + 1 >= max_steps_per_episode:
                        win_status = "DRAW"  # Episode limit reached without decisive victory
                    else:
                        win_status = "LOST"
            
            print(f"Episode {e_idx + 1} finished. Steps: {step + 1}. Reward: {episode_reward:.2f}. Status: {win_status}")
            total_rewards.append(episode_reward)
            battle_wins.append(battle_won)  # Track actual battle outcome

        print("-" * 30)
        print(f"{agent.__class__.__name__} test finished.")
        if total_rewards:
            print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f} (Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f})")
        
        # Print win rate using actual battle outcomes for consistency
        actual_win_count = sum(battle_wins)
        if episodes > 0:
            print(f"Win rate: {actual_win_count/episodes:.2%} ({actual_win_count}/{episodes})")

        # Save episode rewards data for legacy function (single agent testing)
        if hasattr(agent, 'llm_type'):
            llm_type = agent.llm_type
            if llm_type == "llama3":
                config_name = "baseline_pure_llm_pretrained"
            elif llm_type == "ours":
                config_name = "baseline_pure_llm_finetuned"
            elif llm_type == "random":
                config_name = "baseline_pure_llm_random"
            else:
                config_name = f"baseline_pure_llm_{llm_type}"
        elif agent.__class__.__name__ == "RandomAgent":
            config_name = "baseline_pure_random"
            llm_type = "random"
        else:
            config_name = "baseline_unknown_agent"
            llm_type = "unknown"
        
        save_episode_rewards(config_name, total_rewards, episodes, 0.0, llm_type, "none", battle_wins)

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2, SMAC, and dependencies are correctly installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")


def parse_args():
    """Parse command line arguments for ablation study."""
    parser = argparse.ArgumentParser(description='QMIX-LLM Ablation Study')
    # LLM configuration
    parser.add_argument('--llm', type=str, default='none',
                       choices=['llama3', 'ours', 'random', 'none'],
                       help='LLM type: llama3 (pretrained), ours (fine-tuned), random, none')
    
    # Algorithm configuration  
    parser.add_argument('--algo', type=str, default='qmix',
                       choices=['qmix', 'none'],
                       help='Algorithm: qmix or none (pure LLM execution)')
    
    # Alignment weight for QMIX training
    parser.add_argument('--alignment-weight', type=float, default=0.5,
                       help='Weight for alignment reward in QMIX training')
    
    # Environment settings
    parser.add_argument('--map', type=str, default='3m',
                       help='SMAC map name')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per episode')
    
    # Experiment settings
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--save-replay', action='store_true',
                       help='Save game replays')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    
    # Training/Testing mode settings
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='Mode: train (save weights every 50 episodes) or test (load specific weights)')
    parser.add_argument('--load-episode', type=int, default=None,
                       help='Episode number to load weights from (only used in test mode)')
    
    return parser.parse_args()

def create_agents(args):
    """Create agents based on configuration."""
    llm_agent = None
    qmix_agent = None
    
    # Create LLM agent based on type
    if args.llm != 'none':
        if args.llm == 'llama3':
            model_name = "llama3:latest"
        elif args.llm == 'ours':
            model_name = "remijang/smac-sft-gemma3"  # Your fine-tuned model
        elif args.llm == 'random':
            model_name = "random"  # Special case for random LLM
        
        llm_agent = LLMAgent(model_name=model_name, llm_type=args.llm, verbose=args.verbose)
    
    # Create QMIX agent if needed
    if args.algo == 'qmix':
        qmix_agent = QMIXAgent(verbose=args.verbose)
    
    return llm_agent, qmix_agent

def run_experiment(args):
    """Run the experiment based on configuration."""
    print("=" * 60)
    print("QMIX-LLM ABLATION STUDY")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  LLM Type: {args.llm}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Alignment Weight: {args.alignment_weight}")
    print(f"  Map: {args.map}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Mode: {args.mode}")
    print(f"  Render: {args.render}")
    if args.mode == 'test' and args.load_episode:
        print(f"  Load Episode: {args.load_episode}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    llm_agent, qmix_agent = create_agents(args)
    
    # Determine config name for weight saving/loading
    config_name = None
    if args.algo == 'qmix':
        if args.llm == 'none':
            config_name = "baseline_pure_qmix"
        elif args.llm == 'llama3':
            config_name = "qmix_pretrained_llm"
        elif args.llm == 'ours':
            if args.alignment_weight == 0.0:
                config_name = "qmix_finetuned_llm_no_alignment"
            else:
                config_name = "qmix_finetuned_llm"
        elif args.llm == 'random':
            if args.alignment_weight == 0.0:
                config_name = "qmix_random_llm_no_alignment"
            else:
                config_name = "qmix_random_llm"
        else:
            config_name = f"qmix_{args.llm}_llm"
    
    # Determine experiment type and run accordingly
    if args.algo == 'none':
        # Pure LLM execution (baselines) - no weight saving/loading, no training required
        print("🚀 PURE LLM EXECUTION MODE")
        print("ℹ️  No training or weight operations will be performed.")
        print("ℹ️  Mode parameter (train/test) is ignored for pure LLM execution.")
        print("ℹ️  Only performance logging will be saved.\n")
        
        if args.llm == 'none':
            # Pure random agent
            agent = RandomAgent(verbose=args.verbose)
            print("🎲 Running Pure Random Agent (Baseline)...")
        else:
            # Pure LLM execution
            agent = llm_agent
            print(f"🤖 Running Pure LLM Execution ({args.llm.upper()})...")
        
        run_smac_with_agent(
            agent=agent,
            map_name=args.map,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            render=args.render,
            verbose_env=args.verbose,
            save_replay=args.save_replay
        )
    
    elif args.algo == 'qmix':
        print("🔬 QMIX TRAINING/TESTING MODE")
        print("ℹ️  QMIX agent will be trained/tested with weight operations.")
        if args.llm != 'none':
            print(f"ℹ️  LLM guidance ({args.llm.upper()}) with alignment weight: {args.alignment_weight}")
        print(f"ℹ️  Mode: {args.mode.upper()}")
        if args.mode == 'test' and args.load_episode:
            print(f"ℹ️  Will load weights from episode: {args.load_episode}")
        print()
        
        if args.llm == 'none':
            # Pure QMIX (no LLM guidance)
            print("⚔️  Running Pure QMIX (no LLM guidance)...")
            # Use dummy LLM agent that returns no-op actions
            dummy_llm = LLMAgent(llm_type='none', verbose=args.verbose)
            run_smac_with_agents(
                llm_agent=dummy_llm,
                qmix_agent=qmix_agent,
                map_name=args.map,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                verbose_env=args.verbose,
                render=args.render,
                save_replay=args.save_replay,
                alignment_weight=0.0,  # No alignment reward for pure QMIX
                mode=args.mode,
                load_episode=args.load_episode,
                config_name=config_name
            )
        else:
            # QMIX with LLM guidance
            print(f"🤝 Running QMIX + {args.llm.upper()} LLM Guidance...")
            run_smac_with_agents(
                llm_agent=llm_agent,
                qmix_agent=qmix_agent,
                map_name=args.map,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                verbose_env=args.verbose,
                render=args.render,
                save_replay=args.save_replay,
                alignment_weight=args.alignment_weight,
                mode=args.mode,
                load_episode=args.load_episode,
                config_name=config_name
            )


def save_episode_rewards(config_name, episode_rewards, total_episodes, alignment_weight, llm_type, algo_type, battle_wins=None):
    """Save episode rewards data to ablation_results directory organized by configuration."""
    import os
    from datetime import datetime
    
    # Create ablation_results main directory if it doesn't exist
    os.makedirs("ablation_results", exist_ok=True)
    
    # Create config-specific directory within ablation_results
    config_dir = f"ablation_results/{config_name}"
    os.makedirs(config_dir, exist_ok=True)
    
    # Calculate win rate if battle_wins data is provided
    win_rate = 0.0
    if battle_wins and len(battle_wins) > 0:
        win_rate = sum(battle_wins) / len(battle_wins)
    
    # Create data structure
    reward_data = {
        "config_name": config_name,
        "episode_rewards": episode_rewards,
        "total_episodes": total_episodes,
        "alignment_weight": alignment_weight,
        "llm_type": llm_type,
        "algo_type": algo_type,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "average_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
        "min_reward": min(episode_rewards) if episode_rewards else 0,
        "max_reward": max(episode_rewards) if episode_rewards else 0,
        "win_rate": win_rate,
        "total_wins": sum(battle_wins) if battle_wins else 0,
        "battle_outcomes": battle_wins if battle_wins else []
    }
    
    # Save to config-specific directory
    file_path = f"{config_dir}/episode_rewards.json"
    with open(file_path, 'w') as f:
        json.dump(reward_data, f, indent=2)
    
    print(f"📊 Episode rewards saved to: {file_path}")
    return file_path

# GPU Monitoring and Optimization Functions
def check_gpu_utilization():
    """Check current GPU utilization and memory usage."""
    if not torch.cuda.is_available():
        return None
    
    try:
        import subprocess
        import re
        
        # Get GPU stats using nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                parts = lines[0].split(', ')
                utilization = int(parts[0])
                memory_used = int(parts[1])
                memory_total = int(parts[2])
                
                return {
                    'utilization_percent': utilization,
                    'memory_used_mb': memory_used,
                    'memory_total_mb': memory_total,
                    'allocated_gb': memory_used / 1024,
                    'total_gb': memory_total / 1024
                }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass
    
    # Fallback to PyTorch memory info only
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            'utilization_percent': 0,  # Can't get utilization without nvidia-smi
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    except:
        return None

def warm_up_gpu():
    """Warm up GPU with dummy computations to improve initial performance."""
    if not torch.cuda.is_available():
        return
    
    print("🔥 Warming up GPU...")
    device = torch.device("cuda")
    
    # Create dummy tensors and perform computations
    for _ in range(5):
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        z = torch.relu(z)
        
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("✅ GPU warmed up successfully!")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
