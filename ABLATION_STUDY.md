# QMIX-LLM Ablation Study

This ablation study evaluates the effectiveness of LLM guidance in multi-agent reinforcement learning using QMIX on StarCraft II scenarios.

## Experiment Categories

### 1. Baselines (基線)

#### Pure QMIX (無 LLM 指導)
```bash
python test_llm.py --llm none --algo qmix --alignment-weight 0.0
```
- **Purpose**: Basic QMIX without LLM guidance
- **Expected**: Standard QMIX performance baseline

#### Pure LLM Execution (純 LLM 執行)
```bash
# Pre-trained LLM direct execution
python test_llm.py --llm llama3 --algo none

# Fine-tuned LLM direct execution  
python test_llm.py --llm ours --algo none
```
- **Purpose**: LLM as direct policy (expert upper bound reference)
- **Expected**: Shows LLM's raw strategic capability

#### Pure Random Agent
```bash
python test_llm.py --llm none --algo none
```
- **Purpose**: Lower bound baseline for comparison
- **Expected**: Worst performance, random behavior

### 2. LLM Quality & Adaptability Ablations

#### QMIX + Pre-trained LLM Guidance
```bash
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.1
```
- **Purpose**: Current implementation with general LLM
- **Expected**: Moderate improvement over pure QMIX

#### QMIX + Fine-tuned LLM Guidance (核心創新點)
```bash
python test_llm.py --llm ours --algo qmix --alignment-weight 0.1
```
- **Purpose**: **Main experimental group** - domain-specific LLM guidance
- **Expected**: **Best performance** due to specialized knowledge

#### QMIX + Random LLM Guidance
```bash
python test_llm.py --llm random --algo qmix --alignment-weight 0.1
```
- **Purpose**: Control for LLM "intelligence" vs. additional signal
- **Expected**: Minimal improvement, proves LLM quality matters

### 3. Alignment Mechanism Ablations

#### Alignment Weight Sensitivity Analysis
```bash
# Low alignment weight
python test_llm.py --llm ours --algo qmix --alignment-weight 0.05

# Standard alignment weight  
python test_llm.py --llm ours --algo qmix --alignment-weight 0.1

# High alignment weight
python test_llm.py --llm ours --algo qmix --alignment-weight 0.5

# Very high alignment weight
python test_llm.py --llm ours --algo qmix --alignment-weight 1.0
```
- **Purpose**: Find optimal balance between LLM guidance and environment reward
- **Expected**: Performance peaks at intermediate values, drops at extremes

## Quick Usage Examples

### Environment Setup
```bash
# IMPORTANT: Always use the pysc2-env environment
cd /root/DRL-Final

# Method 1: If conda activate works
conda activate /root/miniconda3/envs/pysc2-env

# Method 2: Use absolute path (recommended)
/root/miniconda3/envs/pysc2-env/bin/python test_llm.py [arguments]
```

### Individual Experiments
```bash
# Test a specific configuration
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.1 --episodes 10

# With verbose logging
python test_llm.py --llm ours --algo qmix --alignment-weight 0.1 --verbose --episodes 5

# Different map
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.1 --map 8m --episodes 10
```

### Full Ablation Study
```bash
# Run all experiments automatically
python run_ablation_study.py

# Results will be saved to ablation_results_TIMESTAMP/
```

### Using the Test Helper
```bash
# Quick test with helper script
python test_config.py --llm llama3 --algo qmix --alignment-weight 0.1 --episodes 5
```

## Command Line Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--llm` | `llama3`, `ours`, `random`, `none` | `llama3` | LLM type for guidance |
| `--algo` | `qmix`, `none` | `qmix` | Algorithm choice |
| `--alignment-weight` | float | `0.1` | Weight for alignment reward |
| `--map` | string | `3m` | SMAC map name |
| `--episodes` | int | `10` | Number of episodes |
| `--max-steps` | int | `100` | Max steps per episode |
| `--verbose` | flag | `False` | Enable detailed logging |
| `--render` | flag | `False` | Render environment |
| `--save-replay` | flag | `False` | Save game replays |
| `--seed` | int | `0` | Random seed |

## Expected Results Hierarchy

1. **QMIX + Fine-tuned LLM** (best performance)
2. **QMIX + Pre-trained LLM** (good performance)  
3. **Pure Fine-tuned LLM** (expert reference)
4. **Pure QMIX** (baseline)
5. **QMIX + Random LLM** (minimal improvement)
6. **Pure Pre-trained LLM** (variable)
7. **Pure Random Agent** (worst performance)

## Key Metrics to Compare

- **Environment Reward**: Direct game performance
- **Alignment Reward**: How often QMIX follows LLM suggestions
- **Win Rate**: Percentage of games won
- **Training Stability**: Consistency across episodes
- **Convergence Speed**: How quickly performance improves

## Notes

- The `ours` LLM type uses the `gemma3:4b-it-qat` model
- Adjust the model name in the code if you want to use a different model
- Increase `--episodes` for more robust statistical results (recommended: 50-100)
- Use `--verbose` flag for debugging individual experiments
- Results are saved with timestamps to prevent overwriting
