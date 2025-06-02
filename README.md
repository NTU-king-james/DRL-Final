# QMIX-LLM: LLM-Guided Multi-Agent Reinforcement Learning

![Version](https://img.shields.io/badge/version-1.0-blue)
![Status](https://img.shields.io/badge/status-completed-success)

A comprehensive framework for evaluating Large Language Model (LLM) guidance in Multi-Agent Reinforcement Learning using QMIX on StarCraft II micromanagement scenarios.

## 🎯 Project Overview

This project implements and evaluates a novel QMIX+LLM collaborative framework where LLM strategic guidance is combined with QMIX reinforcement learning for improved multi-agent coordination. The system includes a full ablation study setup with comprehensive visualization and analysis tools.

### Key Research Questions

1. **Baseline Performance**: How much does LLM guidance improve over pure QMIX?
2. **LLM Quality Impact**: Does fine-tuning improve guidance effectiveness?
3. **Alignment Mechanism**: What's the optimal balance between LLM guidance and environment reward?
4. **Training Stability**: Does LLM guidance improve or hurt QMIX convergence?

## 🛠️ Setup & Installation

### Environment Requirements

```bash
# Clone repository
git clone https://github.com/yourusername/qmix-llm.git
cd qmix-llm

# Setup environment
conda create -n pysc2-env python=3.8
conda activate pysc2-env

# Install requirements
pip install -r requirements.txt

# Install StarCraft II dependencies, in pymarl2
bash install_sc2.sh
bash install_dependencies.sh
```

### Map Installation

1. Install PySC2
2. Install SMAC
3. Install SC2
4. Get SMAC maps and put it in SC2PATH/Maps

## 🧪 Experiment Categories

### 1. Baselines

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

### 2. LLM Quality & Adaptability Ablations (Fixed alignment_weight=0.5)

#### QMIX + Pre-trained LLM Guidance

```bash
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.5
```

- **Purpose**: Current implementation with general LLM
- **Expected**: Moderate improvement over pure QMIX

#### QMIX + Fine-tuned LLM Guidance (核心創新點)

```bash
python test_llm.py --llm ours --algo qmix --alignment-weight 0.5
```

- **Purpose**: **Main experimental group** - domain-specific LLM guidance
- **Expected**: **Best performance** due to specialized knowledge

#### QMIX + Random LLM Guidance

```bash
python test_llm.py --llm random --algo qmix --alignment-weight 0.5
```

- **Purpose**: Control for LLM "intelligence" vs. additional signal
- **Expected**: Minimal improvement, proves LLM quality matters

## 🚀 Usage Examples

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
# Test a specific configuration (using fixed alignment weight 0.5)
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.5 --episodes 10

# With verbose logging
python test_llm.py --llm ours --algo qmix --alignment-weight 0.5 --verbose --episodes 5

# Different map
python test_llm.py --llm llama3 --algo qmix --alignment-weight 0.5 --map 8m --episodes 10
```

### Full Ablation Study

```bash
# Run all experiments automatically
python run_full_ablation.py

# Results will be saved to ablation_results_TIMESTAMP/
```

## 📊 Results & Visualization

The project includes a comprehensive visualization and analysis system:

### Analysis Functions

```bash
# Standard analysis
python analyze_results.py

# Comprehensive analysis with plots
python analyze_results.py --comprehensive

# Generate plots only
python analyze_results.py --plots-only

# Analyze specific directory
python analyze_results.py --results-dir /path/to/results --comprehensive
```

### Visualization Outputs

The system generates 4 high-quality research plots:

1. **Baseline Comparison**: QMIX learning curve vs pure LLM performance lines
2. **LLM Quality Comparison**: Multi-curve comparison of different LLM guidance types
3. **Alignment Weight Sensitivity**: Final performance vs alignment weight analysis
4. **Performance Comparison**: Overall performance ranking bar chart

## 📄 Command Line Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--llm` | `llama3`, `ours`, `random`, `none` | `llama3` | LLM type for guidance |
| `--algo` | `qmix`, `none` | `qmix` | Algorithm choice |
| `--alignment-weight` | float | `0.5` | Weight for alignment reward |
| `--map` | string | `3m` | SMAC map name |
| `--episodes` | int | `10` | Number of episodes |
| `--max-steps` | int | `100` | Max steps per episode |
| `--verbose` | flag | `False` | Enable detailed logging |
| `--render` | flag | `False` | Render environment |
| `--save-replay` | flag | `False` | Save game replays |
| `--seed` | int | `0` | Random seed |

## 📝 Expected Results Hierarchy

1. **QMIX + Fine-tuned LLM** (best performance)
2. **QMIX + Pre-trained LLM** (good performance)  
3. **Pure Fine-tuned LLM** (expert reference)
4. **Pure QMIX** (baseline)
5. **QMIX + Random LLM** (minimal improvement)
6. **Pure Pre-trained LLM** (variable)

## 🔗 System Architecture

```
test_llm.py (Main Implementation)
├── LLMAgent (llama3/ours/random/none)
├── QMIXAgent (with alignment reward mechanism)
├── RandomAgent (baseline)
└── run_smac_with_agents() (collaboration framework)

run_full_ablation.py (Experiment Runner)
├── 10 pre-configured experiments
├── Result saving (JSON + logs)
└── Error handling & timeouts

analyze_results.py (Analysis & Visualization Tool)
├── Log parsing for metrics
├── Summary generation  
├── Best configuration identification
├── Four core research plots
├── Comprehensive statistical analysis
├── Demo data generation
└── Multiple analysis modes
```

## 📊 Latest Visualization Results

The complete ablation study has been visualized with the 170-episode comparison showing clear performance advantages of the QMIX+LLM approach. View the generated plots in the `ablation_results` directory.

## 📚 Notes

- The `ours` LLM type uses the `remijang/smac-sft-gemma3` model
- Adjust the model name in the code if you want to use a different model
- Increase `--episodes` for more robust statistical results (recommended: 50-100)
- Use `--verbose` flag for debugging individual experiments
- Results are saved with timestamps to prevent overwriting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyMARL2 framework for QMIX implementation
- SMAC for multi-agent StarCraft II environment
- Meta AI for Llama 3 models
