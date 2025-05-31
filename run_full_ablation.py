#!/usr/bin/env python3
"""
Comprehensive Ablation Study Runner for QMIX-LLM System

This script runs all the ablation study experiments as outlined:
1. Baselines
2. LLM Quality & Adaptability Ablations  
3. Alignment Mechanism Ablations
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime

# Configuration for ablation study
EXPERIMENT_CONFIGS = {
    # 1. Baselines
    "baseline_pure_qmix": {
        "llm": "none", 
        "algo": "qmix", 
        "alignment_weight": 0.0,
        "description": "Pure QMIX (no LLM guidance)"
    },
    "baseline_pure_llm_pretrained": {
        "llm": "llama3", 
        "algo": "none", 
        "alignment_weight": 0.0,
        "description": "Pure LLM Execution (pre-trained llama3:latest)"
    },
    "baseline_pure_llm_finetuned": {
        "llm": "ours", 
        "algo": "none", 
        "alignment_weight": 0.0,
        "description": "Pure LLM Execution (fine-tuned llama3)"
    },
    
    # 2. LLM Quality & Adaptability Ablations
    "qmix_pretrained_llm": {
        "llm": "llama3", 
        "algo": "qmix", 
        "alignment_weight": 0.1,
        "description": "QMIX + Pre-trained LLM Guidance"
    },
    "qmix_random_llm": {
        "llm": "random", 
        "algo": "qmix", 
        "alignment_weight": 0.1,
        "description": "QMIX + Random LLM Guidance"
    },
    "qmix_finetuned_llm": {
        "llm": "ours", 
        "algo": "qmix", 
        "alignment_weight": 0.1,
        "description": "QMIX + Fine-tuned LLM Guidance (MAIN EXPERIMENT)"
    },
    
    # 3. Alignment Weight Sensitivity Analysis (for best configuration)
    "alignment_weight_0.05": {
        "llm": "ours", 
        "algo": "qmix", 
        "alignment_weight": 0.05,
        "description": "QMIX + Fine-tuned LLM (weight=0.05)"
    },
    "alignment_weight_0.5": {
        "llm": "ours", 
        "algo": "qmix", 
        "alignment_weight": 0.5,
        "description": "QMIX + Fine-tuned LLM (weight=0.5)"
    },
    "alignment_weight_1.0": {
        "llm": "ours", 
        "algo": "qmix", 
        "alignment_weight": 1.0,
        "description": "QMIX + Fine-tuned LLM (weight=1.0)"
    },
}

def run_single_experiment(config_name, config, episodes=10, max_steps=100, map_name="3m", seed=42):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    # Create results directory
    results_dir = f"/root/DRL-Final/ablation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Remove timestamp to allow overwriting previous results
    result_file = f"{results_dir}/{config_name}.json"
    
    cmd = [
        "/root/miniconda3/envs/pysc2-env/bin/python",
        "test_llm.py",
        "--llm", config["llm"],
        "--algo", config["algo"],
        "--alignment-weight", str(config["alignment_weight"]),
        "--map", map_name,
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--seed", str(seed),
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Metadata will be saved to: {result_file}")
    
    start_time = time.time()
    
    try:
        # Run command with output directly to terminal (no redirection)
        print(f"\n🎬 Starting experiment: {config_name}")
        print(f"⏱️  Start time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        result = subprocess.run(
            cmd, 
            cwd="/root/DRL-Final", 
            text=True, 
            timeout=1800  # 30 minutes timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 60)
        print(f"🏁 Experiment completed: {config_name}")
        print(f"⏱️  End time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⌛ Duration: {duration:.1f}s")
        
        # Save experiment metadata (without the full log content)
        experiment_result = {
            "config_name": config_name,
            "config": config,
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "duration_seconds": duration,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": result.returncode == 0
        }
        
        with open(result_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - {config_name} completed successfully!")
        else:
            print(f"❌ FAILED - {config_name} failed with return code: {result.returncode}")
        
        print("=" * 80)
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print("-" * 60)
        print(f"⏰ TIMEOUT - {config_name} exceeded {duration:.1f}s limit")
        print("=" * 80)
        
        # Save timeout result
        experiment_result = {
            "config_name": config_name,
            "config": config,
            "command": " ".join(cmd),
            "return_code": -1,
            "duration_seconds": duration,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": "Timeout after 30 minutes"
        }
        
        with open(result_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        return False
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print("-" * 60)
        print(f"💥 ERROR - {config_name} failed with exception: {e}")
        print(f"⌛ Duration: {duration:.1f}s")
        print("=" * 80)
        
        # Save error result
        experiment_result = {
            "config_name": config_name,
            "config": config,
            "command": " ".join(cmd),
            "return_code": -2,
            "duration_seconds": duration,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": str(e)
        }
        
        with open(result_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        return False

def run_ablation_study(episodes=10, max_steps=100, map_name="3m", seed=42, skip_existing=True):
    """Run the complete ablation study."""
    print("🚀 STARTING COMPREHENSIVE ABLATION STUDY")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"   Episodes per experiment: {episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Map: {map_name}")
    print(f"   Random seed: {seed}")
    print(f"🔬 Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print("=" * 80)
    
    results = {}
    successful = 0
    start_time = time.time()
    
    for i, (config_name, config) in enumerate(EXPERIMENT_CONFIGS.items(), 1):
        print(f"\n📈 PROGRESS: [{i}/{len(EXPERIMENT_CONFIGS)}] Starting next experiment...")
        print(f"🔄 Remaining: {len(EXPERIMENT_CONFIGS) - i} experiments")
        
        success = run_single_experiment(
            config_name, 
            config, 
            episodes=episodes, 
            max_steps=max_steps, 
            map_name=map_name, 
            seed=seed
        )
        
        results[config_name] = success
        if success:
            successful += 1
            
        # Show running summary
        current_success_rate = successful / i * 100
        print(f"📊 Running Summary: {successful}/{i} successful ({current_success_rate:.1f}%)")
    
    total_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🏁 ABLATION STUDY COMPLETED")
    print(f"{'='*80}")
    print(f"⏱️  Total Duration: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"📊 Success Rate: {successful}/{len(EXPERIMENT_CONFIGS)} ({successful/len(EXPERIMENT_CONFIGS)*100:.1f}%)")
    print(f"📁 Metadata saved in: /root/DRL-Final/ablation_results/")
    
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 60)
    for config_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED "
        description = EXPERIMENT_CONFIGS[config_name]["description"]
        print(f"  {status} | {config_name}")
        print(f"           {description}")
        print()
    
    if successful == len(EXPERIMENT_CONFIGS):
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    elif successful > 0:
        print(f"⚠️  {successful} out of {len(EXPERIMENT_CONFIGS)} experiments completed successfully.")
    else:
        print("❌ ALL EXPERIMENTS FAILED!")
    
    print("=" * 80)
    
    return results

def main():
    """Main function with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run QMIX-LLM Ablation Study')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per experiment')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--map', type=str, default='3m', help='SMAC map name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 episodes, 50 steps)')
    parser.add_argument('--single', type=str, help='Run single experiment by config name')
    
    args = parser.parse_args()
    
    if args.quick:
        episodes, max_steps = 2, 50
        print("🏃 Running in QUICK mode")
    else:
        episodes, max_steps = args.episodes, args.max_steps
    
    os.chdir("/root/DRL-Final")
    
    if args.single:
        if args.single in EXPERIMENT_CONFIGS:
            print(f"🎯 Running single experiment: {args.single}")
            config = EXPERIMENT_CONFIGS[args.single]
            success = run_single_experiment(
                args.single, 
                config, 
                episodes=episodes, 
                max_steps=max_steps, 
                map_name=args.map, 
                seed=args.seed
            )
            print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        else:
            print(f"❌ Unknown experiment: {args.single}")
            print(f"Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")
    else:
        run_ablation_study(
            episodes=episodes, 
            max_steps=max_steps, 
            map_name=args.map, 
            seed=args.seed
        )

if __name__ == "__main__":
    main()
