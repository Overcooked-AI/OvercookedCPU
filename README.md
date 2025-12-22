# MAPPO Training for Overcooked-AI 🧑‍🍳🤖

Train Multi-Agent Proximal Policy Optimization (MAPPO) agents on CPU using the Overcooked-AI environment.

## 📁 Project Structure

```
your_project/
├── train_mappo.py              # Main training script
├── overcooked_mappo_env.py     # RLlib environment wrapper
├── custom_layout.py            # Custom playground layouts
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
└── results/                    # Training results (auto-created)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python overcooked_mappo_env.py
```

### 2. Train MAPPO Agents

**Basic training (cramped_room layout):**
```bash
python train_mappo.py
```

**Train on different layout:**
```bash
python train_mappo.py --layout asymmetric_advantages
```

**Train with reward shaping:**
```bash
python train_mappo.py --use-phi --iterations 1000
```

**Custom configuration:**
```bash
python train_mappo.py \
    --layout cramped_room \
    --iterations 500 \
    --workers 8 \
    --checkpoint-freq 20 \
    --local-dir ./my_results
```

### 3. Evaluate Trained Agent

```bash
python train_mappo.py \
    --eval-checkpoint ./results/mappo_cramped_room/PPO_xxx/checkpoint_000500 \
    --layout cramped_room \
    --eval-episodes 20
```

## 🎮 Available Layouts

Built-in Overcooked layouts:
- `cramped_room` - Small kitchen, tight coordination
- `asymmetric_advantages` - Asymmetric roles
- `coordination_ring` - Circular layout
- `forced_coordination` - Requires tight coordination
- `counter_circuit` - Long counter layout

Custom playground layouts (in `custom_layout.py`):
- `playground` - Basic playground
- `playground_medium` - Medium-sized kitchen
- `playground_large` - Large kitchen
- `playground_complex` - Multiple pots and obstacles
- `playground_corridor` - Narrow corridor (tests collision)

## 🔧 Configuration

Edit `config.py` to customize:

### Environment Settings
- `layout_name`: Which layout to use
- `horizon`: Episode length (default: 400 steps)
- `use_phi`: Enable shaped rewards
- `reward_shaping_factor`: Scaling for shaped rewards

### Training Hyperparameters
- `num_workers`: Parallel rollout workers (default: 4)
- `num_envs_per_worker`: Vectorized envs per worker (default: 4)
- `train_batch_size`: Training batch size (default: 4000)
- `lr`: Learning rate (default: 5e-4)
- `gamma`: Discount factor (default: 0.99)
- `lambda`: GAE lambda (default: 0.95)

### Model Architecture
- `fcnet_hiddens`: Hidden layer sizes (default: [256, 256])
- `fcnet_activation`: Activation function (default: "relu")

## 🧠 How MAPPO Works

**Multi-Agent PPO (MAPPO)** is a cooperative MARL algorithm:

1. **Shared Policy**: Both agents use the same neural network weights
2. **Parameter Sharing**: Reduces sample complexity in homogeneous settings
3. **Centralized Training**: Uses full state information during training
4. **Decentralized Execution**: Each agent acts based on local observations

### Key Features:
- ✅ Vectorized observations via `lossless_state_encoding` (fast on CPU)
- ✅ No image rendering needed
- ✅ Cooperative reward: Both agents get same reward
- ✅ Collision detection handled by Overcooked MDP
- ✅ Full game mechanics: pickup, place, cook, deliver

## 📊 Monitoring Training

Ray RLlib provides built-in TensorBoard logging:

```bash
tensorboard --logdir ./results
```

Key metrics to watch:
- `episode_reward_mean`: Average episode return
- `policy_loss`: Policy network loss
- `vf_loss`: Value function loss
- `entropy`: Policy exploration

## 🎯 Expected Performance

| Layout | Steps to Convergence | Final Reward |
|--------|---------------------|--------------|
| cramped_room | ~1-2M | 15-25 |
| asymmetric_advantages | ~2-3M | 20-30 |
| coordination_ring | ~2-4M | 10-20 |

*Performance varies based on hyperparameters and random seed*

## 🔬 Advanced Usage

### Custom Layouts

Create your own layouts in `custom_layout.py`:

```python
MY_LAYOUT = """
XXPXX
O  2D
X   X
D1  S
"""
```

Grid symbols:
- ` ` = Empty floor
- `X` = Counter
- `O` = Onion dispenser
- `T` = Tomato dispenser
- `P` = Pot (cooking)
- `D` = Dish dispenser
- `S` = Serving location
- `1`, `2` = Agent starting positions

### Reward Shaping

Enable potential-based reward shaping for faster learning:

```python
ENV_CONFIG["use_phi"] = True
ENV_CONFIG["reward_shaping_factor"] = 1.0
```

The shaped reward uses Overcooked's built-in potential function which estimates progress toward delivering soups.

### Multi-GPU Training (if available)

Edit `config.py`:
```python
TRAINING_CONFIG["num_gpus"] = 1
```

## 📖 Key Files Explained

### `overcooked_mappo_env.py`
- Wraps `OvercookedEnv` for RLlib
- Uses `lossless_state_encoding()` for observations
- Returns observations as flattened vectors (fast on CPU)
- Handles cooperative reward distribution

### `train_mappo.py`
- Configures PPO as MAPPO (shared policy)
- Manages training loop via Ray Tune
- Handles checkpointing and evaluation

### `config.py`
- Centralizes all hyperparameters
- Easy to modify without touching code

### `custom_layout.py`
- Defines custom Overcooked layouts
- Helper functions to create MDPs from strings

## 🐛 Troubleshooting

**Ray initialization errors:**
```bash
ray stop  # Stop any existing Ray processes
python train_mappo.py
```

**Out of memory:**
- Reduce `num_workers` or `num_envs_per_worker`
- Reduce `train_batch_size`

**Slow training:**
- Increase `num_workers` (up to your CPU cores)
- Increase `num_envs_per_worker` (test for optimal value)

**Agent not learning:**
- Try reward shaping: `--use-phi`
- Increase learning rate
- Increase training duration

## 📚 Citations

If you use this code, please cite:

```bibtex
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={NeurIPS},
  year={2019}
}
```

## 🤝 Contributing

Contributions welcome! Key areas:
- Additional reward shaping functions
- More efficient state encodings
- Curriculum learning strategies
- Human-AI coordination experiments

## 📄 License

This project uses the Overcooked-AI environment (MIT License).

## 🔗 Resources

- [Overcooked-AI GitHub](https://github.com/HumanCompatibleAI/overcooked_ai)
- [Ray RLlib Docs](https://docs.ray.io/en/latest/rllib/index.html)
- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [Original Overcooked Paper](https://arxiv.org/abs/1910.05789)

---

**Happy Training! 🚀**