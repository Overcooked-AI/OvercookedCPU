# config.py
"""
Configuration file for MAPPO training on Overcooked
"""

# ============= ENVIRONMENT CONFIG =============
ENV_CONFIG = {
    "layout_name": "cramped_room",  # Options: cramped_room, asymmetric_advantages, coordination_ring, forced_coordination, counter_circuit
    "horizon": 400,
    "use_phi": False,  # Whether to use shaped rewards (potential function)
    "reward_shaping_horizon": 10000000,
    "reward_shaping_factor": 1.0,
}

# ============= CUSTOM LAYOUT (String Format) =============
# Grid format: X=Counter, O=Onion, T=Tomato, P=Pot, D=Dish, S=Serving
# Numbers (1,2) = Starting positions for agents
CUSTOM_LAYOUT = """
XXPXX
O  2D
X   X
X   X
D1  S
"""

# ============= TRAINING CONFIG =============
TRAINING_CONFIG = {
    # Resources
    "num_gpus": 0,  # CPU training
    "num_workers": 4,  # Parallel envs
    "num_envs_per_worker": 4,  # Vectorized envs per worker
    "num_cpus_per_worker": 1,
    
    # Training
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    
    # Learning
    "lr": 5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "vf_clip_param": 50.0,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.5,
    
    # Rollout
    "rollout_fragment_length": 200,
    "batch_mode": "truncate_episodes",
    
    # Model
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,  # Separate value network
    },
    
    # Evaluation
    "evaluation_interval": 10,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    "evaluation_num_workers": 1,
    
    # Other
    "framework": "torch",
    "log_level": "INFO",
}

# ============= EXPERIMENT CONFIG =============
EXPERIMENT_CONFIG = {
    "name": "mappo_overcooked",
    "checkpoint_freq": 10,
    "checkpoint_at_end": True,
    "stop": {
        "training_iteration": 500,
        "timesteps_total": 2000000,
    },
    "local_dir": "./results",
}