# QMIX-LLM: LLM-Guided Multi-Agent Reinforcement Learning

A comprehensive framework for evaluating Large Language Model (LLM) guidance in Multi-Agent Reinforcement Learning using QMIX on StarCraft II micromanagement scenarios.

## 🎯 Project Overview

This project implements and evaluates a novel QMIX+LLM collaborative framework where LLM strategic guidance is combined with QMIX reinforcement learning for improved multi-agent coordination. The repository includes a complete ablation study setup and a customized training pipeline.

## 🛠️ Setup & Installation

### Environment Requirements

```bash
# Clone repository
git clone https://github.com/NTU-king-james/DRL-Final.git
cd DRL-Final

# Setup environment
conda create -n pysc2-env python=3.8
conda activate pysc2-env

# Install requirements
pip install -r requirements.txt

```
### Ollama Setup
1. Install [Ollama](https://ollama.com/docs/installation)
2. Pull the LLM model:
   ```bash
   ollama pull llama3
   ollama pull remijang/smac-sft-gemma3
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```


### Map Installation

1. Install PySC2
2. Install SMAC
3. Install SC2
4. Get SMAC maps and put it in SC2PATH/Maps

## 🧪 Experiment

### Pure QMIX

```bash
python train.py --llm none --algo qmix --alignment-weight 0.0
```

### Pure LLM Execution

```bash
# Pre-trained LLM direct execution
python train.py --llm llama3 --algo none

# Fine-tuned LLM direct execution  
python train.py --llm ours --algo none
```

### QMIX + Pre-trained LLM Guidance

```bash
python train.py --llm llama3 --algo qmix --alignment-weight 0.5
```

### QMIX + Fine-tuned LLM Guidance

```bash
python train.py --llm ours --algo qmix --alignment-weight 0.5
```

### QMIX + Random LLM Guidance

```bash
python train.py --llm random --algo qmix --alignment-weight 0.5
```

### Full Ablation Study

```bash
# Run all experiments automatically
python run_full_ablation.py
```



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

## 📊 Results

You can see the results of the experiments in the `ablation_results` directory.

## 📚 Notes

- The `ours` LLM type uses the `remijang/smac-sft-gemma3` model
- Adjust the model name in the code if you want to use a different model
- Increase `--episodes` for more robust statistical results (recommended: 100-200)
- Use `--verbose` flag for debugging individual experiments

## 🙏 Acknowledgments

- PyMARL2 framework for Multi-Agent Reinforcement Learning Environment
- SMAC for multi-agent StarCraft II environment
- Meta AI for Llama 3 models