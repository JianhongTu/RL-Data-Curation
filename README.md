# RL-Data-Curation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Reinforcement Learning for Intelligent Data Subset Selection in Large-Scale Instruction Tuning**

---

## 📖 Overview

Large-scale instruction tuning datasets have become fundamental for training powerful language models. However, these datasets often contain redundant, low-quality, or uninformative samples that increase training costs without proportional gains in model performance. Training on all available data is computationally expensive and inefficient.

**RL-Data-Curation** tackles this challenge by framing data selection as a sequential decision-making problem solved through reinforcement learning. Rather than relying on heuristics or greedy approaches, our RL agent learns an intelligent policy that selects the most informative and diverse subset of training data, maximizing downstream model performance while minimizing computational overhead.

## 🎯 Motivation

- **Efficiency**: Large datasets waste compute on redundant samples
- **Quality**: Not all data contributes equally to model performance
- **Diversity**: Representative subsets should cover the full data distribution

## 🔬 Research Questions

This project investigates three core questions:

1. **Reward Formulation**: How can we design reward functions that effectively incorporate both data quality and diversity metrics?

2. **Computational Efficiency**: How does RL-based selection compare in speed to traditional greedy and submodular optimization methods?

3. **Algorithm Simplification**: Can we simplify existing RL algorithms specifically for the data selection task to improve efficiency and interpretability?

## 🏗️ Architecture

The project is organized into the following structure:

```
RL-Data-Curation/
├── src/
│   └── dss/              # Data Subset Selection
│       ├── env/          # RL Environment definitions
│       └── rl/           # RL algorithms and policies
├── data/                 # Dataset handling and preprocessing
└── configs/             # Configuration files
```

### Key Components

- **Environment (`src/dss/env/`)**: Defines the sequential decision process for data selection, including state representation, action space, and reward computation.

- **RL Algorithms (`src/dss/rl/`)**: Implementations of RL algorithms adapted for data selection, including policy networks and training loops.

## 🔍 Methodology

### Problem Formulation

We model data selection as a Markov Decision Process (MDP):

- **State**: Representation of currently selected data
- **Action**: Binary decision to include or exclude a data sample
- **Reward**: Composite metric incorporating:
  - **Quality**: Proxy metrics for sample informativeness
  - **Diversity**: Coverage of the data distribution
  - **Downstream Performance**: Validation metrics from fine-tuned models

### Approach

1. **Sequential Selection**: The agent iteratively selects samples from the candidate pool
2. **Policy Learning**: Through trial and error, the agent learns which samples contribute most to model performance
3. **Generalization**: Trained policies can transfer to new datasets or tasks

## 📊 Baselines & Comparisons

We compare our RL-based approach against:

- **Random Sampling**: Baseline for comparison
- **Heuristic Methods**: Quality-based scoring (perplexity, loss, etc.)
- **Submodular Optimization**: Greedy maximization of submodular quality functions
- **Diversity Sampling**: k-means clustering, core-set selection

## 📚 Related Work

This project builds upon recent advances in data selection and curriculum learning:

- [**Less Is More for Alignment (LIMA)**](https://arxiv.org/abs/2305.11206) - Demonstrates that high-quality small datasets can match large-scale training
- [**AlpaGasus**](https://arxiv.org/abs/2308.12067) - Selective instruction tuning using LLM-based filtering
- [**Active Instruction Tuning**](https://arxiv.org/abs/2402.04333) - Active learning approaches for instruction data selection
