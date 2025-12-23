# MAPPO Training for Overcooked-AI 🧑‍🍳🤖

Train Multi-Agent Proximal Policy Optimization (MAPPO) agents on CPU using the Overcooked-AI environment with advanced features including curriculum learning, adaptive learning rates, and comprehensive behavioral analysis.

## ✨ Current Features

### 🎯 Core Training
- ✅ **MAPPO Implementation** - Shared policy with centralized value function
- ✅ **CPU-Optimized** - Vectorized observations, no rendering required
- ✅ **Padded Observations** - Transfer learning across different map sizes
- ✅ **Reward Shaping** - Optional potential-based shaped rewards
- ✅ **Multiple Layouts** - 5 built-in + 5 custom playground layouts

### 📊 Advanced Training Features
- ✅ **Curriculum Learning** - Progressive difficulty scheduling
  - Progressive: Easy → Medium → Hard
  - Mixed: Random difficulty sampling
  - Reverse: Hard → Easy (robustness training)
- ✅ **Adaptive Learning Rate** - Dynamic LR adjustment
  - Cosine annealing
  - Step decay
  - Plateau detection
- ✅ **Hyperparameter Optimization** - Ray Tune integration
  - ASHA scheduler for early stopping
  - Population-Based Training (PBT)
  - Optuna search algorithm
- ✅ **Multi-Layout Training** - Single policy across multiple maps

### 🔬 Monitoring & Analysis
- ✅ **Real-Time Monitoring** - TensorBoard integration
  - Core metrics (reward, length, losses)
  - Policy metrics (entropy, KL divergence)
  - Resource metrics (sample/learn time)
- ✅ **Coordination Metrics** - Specialized MARL metrics
  - Task specialization (who carries what)
  - Collision frequency
  - Handoff coordination
  - Action synchronization
  - Spatial coverage & overlap
  - Steps per soup efficiency
- ✅ **Behavioral Analysis** - Comprehensive post-training analysis
  - Position heatmaps
  - Action distribution analysis
  - Task specialization visualization
  - Automated report generation
- ✅ **Live Training Monitor** - Real-time console dashboard

### 🛠️ Developer Tools
- ✅ **Checkpoint Management** - Save/load/evaluate models
- ✅ **Custom Callbacks** - Extensible training hooks
- ✅ **Progress Tracking** - Rolling statistics (100 episodes)
- ✅ **Visualization Suite** - Matplotlib/Seaborn plotting
- ✅ **Error Handling** - Robust fallbacks and warnings

## 🚀 Future Features

### 🎮 Gameplay & Interaction
- 🔲 **Human-AI Interface** - Web-based interface for human players
  - Real-time gameplay with trained agents
  - Action probability visualization
  - Intent prediction display
- 🔲 **Replay System** - Record and playback episodes
  - Video generation from trajectories
  - State-by-state inspection
- 🔲 **Multi-Agent Scenarios** - Beyond 2-player
  - 3-4 agent coordination
  - Asymmetric teams

### 🤖 Advanced Algorithms
- 🔲 **Value Decomposition** - QMIX, QTRAN implementations
- 🔲 **Communication Protocols** - Learned agent communication
  - CommNet, TarMAC architectures
- 🔲 **Hierarchical RL** - High-level goal + low-level actions
- 🔲 **Meta-Learning** - Few-shot adaptation to new layouts
- 🔲 **Inverse RL** - Learn from human demonstrations

### 📈 Training Enhancements
- 🔲 **Prioritized Experience Replay** - Better sample efficiency
- 🔲 **Hindsight Experience Replay** - Learn from failures
- 🔲 **Curiosity-Driven Exploration** - Intrinsic motivation
- 🔲 **Self-Play Evolution** - Train against past selves
- 🔲 **Opponent Modeling** - Predict partner behavior
- 🔲 **Multi-Task Learning** - Train on multiple objectives simultaneously

### 🧠 Intelligence Features
- 🔲 **Theory of Mind** - Model partner's beliefs
- 🔲 **Emergent Language** - Agents develop communication
- 🔲 **Concept Learning** - Abstract task representations
- 🔲 **Few-Shot Generalization** - Quick adaptation to new partners

### 🔍 Analysis & Interpretability
- 🔲 **Attention Visualization** - What agents focus on
- 🔲 **Counterfactual Analysis** - "What if" scenarios
- 🔲 **Skill Discovery** - Automatic primitive identification
- 🔲 **Failure Mode Classification** - Automatic bug detection
- 🔲 **Coordination Score** - Quantitative metrics for teamwork

### 🌐 Infrastructure
- 🔲 **Distributed Training** - Multi-node Ray cluster support
- 🔲 **GPU Acceleration** - Optional GPU training
- 🔲 **Cloud Integration** - AWS/GCP deployment scripts
- 🔲 **Model Serving** - REST API for inference
- 🔲 **MLflow Integration** - Experiment tracking
- 🔲 **Weights & Biases** - Advanced logging

### 🎯 Domain Extensions
- 🔲 **Custom Recipes** - More complex cooking mechanics
- 🔲 **Dynamic Obstacles** - Moving hazards
- 🔲 **Stochastic Events** - Random failures, delays
- 🔲 **Resource Constraints** - Limited ingredients, time pressure
- 🔲 **Procedural Generation** - Infinite layout variations

### 🤝 Human-AI Coordination
- 🔲 **Adaptation to Human Style** - Real-time partner modeling
- 🔲 **Legibility** - Making actions interpretable to humans
- 🔲 **Assistive Agents** - Predict and fill human intentions
- 🔲 **Human Data Collection** - Integrated annotation tools
- 🔲 **Preference Learning** - Learn from human feedback

## 📁 Project Structure

```
your_project/
├── train_mappo.py              # Basic MAPPO training
├── adaptive_trainer.py         # Curriculum + adaptive LR
├── tune_hyperparams.py         # Hyperparameter optimization
├── analyze_behavior.py         # Behavioral analysis tools
├── training_monitor.py         # TensorBoard monitoring
├── coordination_metrics.py     # Specialized coordination metrics
├── overcooked_mappo_env.py     # RLlib environment wrapper
├── custom_layout.py            # Custom playground layouts
├── config.py                   # Configuration parameters
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── quickstart.sh              # Quick start script
└── README.md                  # This file
```

## 🚀 Quick Start

### Basic Training
```bash
# Simple training
python train_mappo.py

# With reward shaping
python train_mappo.py --use-phi

# Different layout
python train_mappo.py --layout asymmetric_advantages --iterations 1000
```

### Advanced Training
```bash
# Curriculum learning + adaptive LR
python adaptive_trainer.py \
    --enable-curriculum \
    --enable-adaptive-lr \
    --curriculum-type progressive \
    --lr-strategy cosine_annealing \
    --iterations 1000

# Hyperparameter optimization
python tune_hyperparams.py \
    --search-type medium \
    --num-samples 20 \
    --scheduler asha
```

### Analysis & Monitoring
```bash
# Live monitoring
python training_monitor.py --results-dir ./results

# TensorBoard
tensorboard --logdir ./results

# Behavioral analysis
python analyze_behavior.py \
    --checkpoint ./results/checkpoint_500 \
    --layout cramped_room \
    --episodes 50
```

## 🎮 Available Layouts

**Built-in Layouts:**
- `cramped_room` - Small kitchen, tight coordination
- `asymmetric_advantages` - Asymmetric roles
- `coordination_ring` - Circular layout
- `forced_coordination` - Requires tight coordination
- `counter_circuit` - Long counter layout

**Custom Playground Layouts:**
- `playground` - Basic playground (5x5)
- `playground_medium` - Medium-sized kitchen (7x6)
- `playground_large` - Large kitchen (9x6)
- `playground_complex` - Multiple pots and obstacles (11x6)
- `playground_corridor` - Narrow corridor (7x6)

## 🔧 Configuration

### Environment Settings (`config.py`)
```python
ENV_CONFIG = {
    "layout_name": "cramped_room",
    "horizon": 400,
    "use_phi": False,
    "reward_shaping_factor": 1.0,
}
```

### Training Hyperparameters
```python
TRAINING_CONFIG = {
    "num_workers": 4,              # Parallel environments
    "num_envs_per_worker": 4,      # Vectorized envs
    "train_batch_size": 4000,      # Samples per iteration
    "lr": 5e-4,                    # Learning rate
    "gamma": 0.99,                 # Discount factor
    "entropy_coeff": 0.01,         # Exploration bonus
}
```

## 📊 Monitoring Metrics

### Core Metrics
- **Episode Reward Mean** - Average return per episode
- **Episode Length** - Average steps per episode
- **Policy Loss** - Policy gradient loss
- **Value Loss** - Value function MSE
- **Entropy** - Policy exploration measure
- **KL Divergence** - Policy change magnitude

### Coordination Metrics
- **Soups Delivered** - Task completion count
- **Steps per Soup** - Efficiency measure
- **Collisions per 100 Steps** - Conflict frequency
- **Task Balance** - Work distribution equality
- **Ingredient Gathering Balance** - Role specialization
- **Spatial Overlap** - Shared workspace usage
- **Action Synchronization** - Joint action patterns
- **Complementary Actions** - Move-interact coordination

## 🧠 How MAPPO Works

**Multi-Agent PPO (MAPPO)** achieves coordination through:

1. **Shared Policy**: Both agents use identical network weights
   - Reduces sample complexity
   - Enables symmetric coordination
   - Natural for homogeneous agents

2. **Parameter Sharing**: Single network for all agents
   - Faster learning from pooled experiences
   - Better generalization across scenarios

3. **Centralized Training**: Uses full state during learning
   - Critic sees global information
   - Actor only needs local observations

4. **Decentralized Execution**: Agents act independently
   - No communication required at test time
   - Robust to partial observability

### Key Features
- ✅ Vectorized observations via `lossless_state_encoding`
- ✅ Padded to fixed size (10×10) for transfer learning
- ✅ No image rendering (CPU-efficient)
- ✅ Cooperative reward (both agents get same signal)
- ✅ Full game mechanics (pickup, place, cook, deliver)

## 📈 Expected Performance

| Layout | Training Time | Final Reward | Soups/Episode |
|--------|---------------|--------------|---------------|
| cramped_room | 30-60 min | 15-25 | 3-5 |
| asymmetric_advantages | 45-90 min | 20-30 | 4-6 |
| coordination_ring | 60-120 min | 10-20 | 2-4 |
| forced_coordination | 90-180 min | 15-25 | 3-5 |

*(4 workers, 4 envs/worker on modern CPU)*

## 🔬 Advanced Usage

### Curriculum Learning
```python
# Progressive difficulty
python adaptive_trainer.py \
    --enable-curriculum \
    --curriculum-type progressive

# Reverse curriculum (robustness)
python adaptive_trainer.py \
    --enable-curriculum \
    --curriculum-type reverse
```

### Adaptive Learning Rate
```python
# Cosine annealing
python adaptive_trainer.py \
    --enable-adaptive-lr \
    --lr-strategy cosine_annealing

# Plateau-based reduction
python adaptive_trainer.py \
    --enable-adaptive-lr \
    --lr-strategy plateau
```

### Hyperparameter Optimization
```python
# Quick search (6 params)
python tune_hyperparams.py --search-type quick --num-samples 10

# Medium search (11 params)
python tune_hyperparams.py --search-type medium --num-samples 20

# Full search (16 params)
python tune_hyperparams.py --search-type full --num-samples 50
```

### Behavioral Analysis
```python
# Generate all visualizations
python analyze_behavior.py \
    --checkpoint ./results/checkpoint_500 \
    --episodes 100 \
    --save-dir ./analysis

# Output:
# - Position heatmaps
# - Action distribution plots
# - Task specialization charts
# - Comprehensive text report
```

## 🐛 Troubleshooting

**Ray initialization errors:**
```bash
ray stop  # Stop existing processes
python train_mappo.py
```

**Out of memory:**
- Reduce `num_workers` or `num_envs_per_worker`
- Reduce `train_batch_size`
- Close other applications

**Slow training:**
- Increase `num_workers` (up to CPU cores)
- Increase `num_envs_per_worker` (test 2-8)
- Reduce observation padding size if using small maps only

**Agent not learning:**
- Enable reward shaping: `--use-phi`
- Try curriculum learning
- Increase training duration
- Check TensorBoard: `tensorboard --logdir ./results`

**TensorBoard not showing data:**
```bash
pip install tensorboard
# Check log directory exists
ls results/*/tensorboard/
```

## 📚 Citations

If you use this code, please cite:

```bibtex
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={NeurIPS},
  year={2019}
}

@article{yu2022surprising,
  title={The surprising effectiveness of ppo in cooperative multi-agent games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  journal={NeurIPS},
  year={2022}
}
```

## 🤝 Contributing

Contributions welcome! Key areas:
- Additional reward shaping functions
- More efficient state encodings
- New coordination metrics
- Curriculum learning strategies
- Human-AI interaction experiments

## 📄 License

This project uses the Overcooked-AI environment (MIT License).

## 🔗 Resources

- [Overcooked-AI GitHub](https://github.com/HumanCompatibleAI/overcooked_ai)
- [Ray RLlib Docs](https://docs.ray.io/en/latest/rllib/index.html)
- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [Original Overcooked Paper](https://arxiv.org/abs/1910.05789)

---

**Happy Training! 🚀**