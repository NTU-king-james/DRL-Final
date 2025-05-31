#!/usr/bin/env python3
"""
Analyze ablation study results - Simple episode reward plotting
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_episode_rewards(results_dir="ablation_results"):
    """Load episode reward data from JSON files."""
    reward_data = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist")
        return reward_data
    
    # Look for episode reward JSON files
    for json_file in results_path.glob("*_episode_rewards.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                config_name = data.get('config_name', json_file.stem.replace('_episode_rewards', ''))
                reward_data[config_name] = data
                print(f"✅ Loaded {config_name}: {len(data.get('episode_rewards', []))} episodes")
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    return reward_data

def plot_episode_rewards(reward_data, save_plots=True):
    """Create individual episode reward plots for each configuration."""
    if not reward_data:
        print("No reward data to plot")
        return
    
    # Create plots directory
    if save_plots:
        os.makedirs("plots", exist_ok=True)
    
    # Plot each configuration separately
    for config_name, data in reward_data.items():
        episode_rewards = data.get('episode_rewards', [])
        if not episode_rewards:
            print(f"No episode rewards for {config_name}")
            continue
            
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(episode_rewards) + 1)
        plt.plot(episodes, episode_rewards, 'b-', linewidth=1.5, alpha=0.7)
        
        # Add running average
        window_size = min(10, len(episode_rewards) // 4) if len(episode_rewards) > 4 else 1
        if window_size > 1:
            running_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            avg_episodes = range(window_size, len(episode_rewards) + 1)
            plt.plot(avg_episodes, running_avg, 'r-', linewidth=2, label=f'Running Avg ({window_size} episodes)')
            plt.legend()
        
        plt.xlabel('Episode')
        plt.ylabel('Environment Reward')
        plt.title(f'Episode Rewards: {config_name}')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_reward = data.get('average_reward', np.mean(episode_rewards))
        min_reward = data.get('min_reward', np.min(episode_rewards))
        max_reward = data.get('max_reward', np.max(episode_rewards))
        stats_text = f'Avg: {avg_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
        
        if save_plots:
            filename = f"plots/{config_name}_episode_rewards.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {filename}")
        
        plt.show()

def create_summary_plot(reward_data, save_plot=True):
    """Create a summary plot showing all configurations."""
    if not reward_data:
        print("No reward data for summary plot")
        return
    
    # Expected configurations for ablation study
    expected_configs = [
        'baseline_pure_qmix',
        'baseline_pure_llm_pretrained', 
        'qmix_finetuned_llm',
        'alignment_weight_0.01',
        'alignment_weight_0.1', 
        'alignment_weight_0.5',
        'alignment_weight_1.0',
        'alignment_weight_2.0',
        'alignment_weight_5.0'
    ]
    
    # Filter to available configurations
    available_configs = [config for config in expected_configs if config in reward_data]
    
    if not available_configs:
        print("No expected configurations found for summary plot")
        return
    
    # Create subplot grid
    n_configs = len(available_configs)
    cols = 3
    rows = (n_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, config_name in enumerate(available_configs):
        data = reward_data[config_name]
        episode_rewards = data.get('episode_rewards', [])
        
        if not episode_rewards:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(config_name)
            continue
        
        episodes = range(1, len(episode_rewards) + 1)
        axes[i].plot(episodes, episode_rewards, 'b-', linewidth=1, alpha=0.7)
        
        # Add running average
        window_size = min(10, len(episode_rewards) // 4) if len(episode_rewards) > 4 else 1
        if window_size > 1:
            running_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            avg_episodes = range(window_size, len(episode_rewards) + 1)
            axes[i].plot(avg_episodes, running_avg, 'r-', linewidth=1.5)
        
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Reward')
        axes[i].set_title(config_name)
        axes[i].grid(True, alpha=0.3)
        
        # Add avg reward text
        avg_reward = data.get('average_reward', np.mean(episode_rewards))
        axes[i].text(0.02, 0.98, f'Avg: {avg_reward:.2f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(available_configs), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("plots", exist_ok=True)
        filename = "plots/ablation_summary_episode_rewards.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Saved summary plot: {filename}")
    
    plt.show()

def main():
    """Main function to analyze ablation study results."""
    print("🔍 Loading episode reward data...")
    reward_data = load_episode_rewards()
    
    if not reward_data:
        print("❌ No reward data found. Run experiments first to generate episode reward data.")
        return
    
    print(f"\n📊 Found {len(reward_data)} configurations:")
    for config_name, data in reward_data.items():
        episodes = len(data.get('episode_rewards', []))
        avg_reward = data.get('average_reward', 0)
        print(f"  - {config_name}: {episodes} episodes, avg reward: {avg_reward:.2f}")
    
    print("\n🎯 Creating individual episode reward plots...")
    plot_episode_rewards(reward_data)
    
    print("\n📈 Creating summary plot...")
    create_summary_plot(reward_data)
    
    print("\n✅ Analysis complete! Check the 'plots/' directory for generated figures.")

if __name__ == "__main__":
    main()