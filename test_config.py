#!/usr/bin/env python3
"""
Quick test script for individual ablation study configurations.

Examples:
    # Pure QMIX (baseline)
    python test_config.py --llm none --algo qmix --alignment-weight 0.0
    
    # QMIX with LLM guidance
    python test_config.py --llm llama3 --algo qmix --alignment-weight 0.1
    
    # Pure LLM execution
    python test_config.py --llm llama3 --algo none
    
    # Random baseline
    python test_config.py --llm random --algo qmix --alignment-weight 0.1
"""

import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    # Pass all arguments to test_llm.py
    cmd = [sys.executable, "test_llm.py"] + sys.argv[1:]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
