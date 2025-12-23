# tune_hyperparams.py
"""
Hyperparameter optimization for MAPPO using Ray Tune
Systematically searches for optimal training configuration
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import sys
import os
from pathlib import Path

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from overcooked_mappo_env import make_overcooked_env
from config import ENV_CONFIG
from training_monitor import TrainingMonitor


def create_search_space(search_type: str = "medium"):
    """
    Define hyperparameter search spaces.
    
    Args:
        search_type:
            - "quick": Fast search, fewer parameters
            - "medium": Balanced search
            - "full": Exhaustive search
            
    Returns:
        Dictionary of parameter search spaces
    """
    
    if search_type == "quick":
        return {
            # Learning
            "lr": tune.loguniform(1e-4, 5e-4),
            "gamma": tune.choice([0.99, 0.995]),
            "lambda": tune.choice([0.95, 0.98]),
            
            # Training
            "sgd_minibatch_size": tune.choice([128, 256]),
            "num_sgd_iter": tune.choice([5, 10]),
            
            # Exploration
            "entropy_coeff": tune.loguniform(0.005, 0.02),
        }
    
    elif search_type == "medium":
        return {
            # Learning
            "lr": tune.loguniform(5e-5, 1e-3),
            "gamma": tune.choice([0.95, 0.99, 0.995, 0.999]),
            "lambda": tune.choice([0.9, 0.95, 0.98]),
            
            # PPO specific
            "clip_param": tune.uniform(0.1, 0.3),
            "vf_clip_param": tune.choice([5.0, 10.0, 20.0]),
            
            # Training
            "sgd_minibatch_size": tune.choice([64, 128, 256]),
            "num_sgd_iter": tune.choice([5, 10, 15]),
            "train_batch_size": tune.choice([2000, 4000, 8000]),
            
            # Exploration
            "entropy_coeff": tune.loguniform(0.001, 0.05),
            
            # Network
            "fcnet_hiddens": tune.choice([
                [256, 256],
                [512, 512],
                [256, 256, 256],
            ]),
        }
    
    elif search_type == "full":
        return {
            # Learning
            "lr": tune.loguniform(1e-5, 5e-3),
            "gamma": tune.uniform(0.95, 0.999),
            "lambda": tune.uniform(0.9, 0.99),
            
            # PPO specific
            "clip_param": tune.uniform(0.05, 0.4),
            "vf_clip_param": tune.uniform(1.0, 50.0),
            "vf_loss_coeff": tune.uniform(0.3, 1.0),
            
            # Training
            "sgd_minibatch_size": tune.choice([32, 64, 128, 256, 512]),
            "num_sgd_iter": tune.randint(3, 20),
            "train_batch_size": tune.choice([1000, 2000, 4000, 8000, 16000]),
            
            # Exploration
            "entropy_coeff": tune.loguniform(0.0001, 0.1),
            
            # Network
            "fcnet_hiddens": tune.choice([
                [128, 128],
                [256, 256],
                [512, 512],
                [256, 256, 256],
                [512, 512, 512],
            ]),
            "fcnet_activation": tune.choice(["relu", "tanh"]),
            
            # Reward shaping
            "use_phi": tune.choice([True, False]),
            "reward_shaping_factor": tune.uniform(0.5, 2.0),
        }
    
    return {}


def build_config_with_params(params: dict, base_env_config: dict):
    """
    Build PPO config with sampled hyperparameters.
    
    Args:
        params: Sampled hyperparameters from search space
        base_env_config: Base environment configuration
        
    Returns:
        PPOConfig instance
    """
    
    # Extract environment params
    env_config = base_env_config.copy()
    if "use_phi" in params:
        env_config["use_phi"] = params["use_phi"]
    if "reward_shaping_factor" in params:
        env_config["reward_shaping_factor"] = params["reward_shaping_factor"]
    
    # Extract network params
    model_config = {
        "fcnet_hiddens": params.get("fcnet_hiddens", [256, 256]),
        "fcnet_activation": params.get("fcnet_activation", "relu"),
        "vf_share_layers": False,
    }
    
    # Build config
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            env="overcooked_mappo",
            env_config=env_config,
        )
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=params.get("num_workers", 4),
            num_envs_per_env_runner=params.get("num_envs_per_worker", 4),
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
            num_cpus_per_env_runner=1,
        )
        .training(
            train_batch_size=params.get("train_batch_size", 4000),
            minibatch_size=params.get("sgd_minibatch_size", 128),
            num_sgd_iter=params.get("num_sgd_iter", 10),
            lr=params.get("lr", 5e-4),
            gamma=params.get("gamma", 0.99),
            lambda_=params.get("lambda", 0.95),
            clip_param=params.get("clip_param", 0.2),
            vf_clip_param=params.get("vf_clip_param", 10.0),
            entropy_coeff=params.get("entropy_coeff", 0.01),
            vf_loss_coeff=params.get("vf_loss_coeff", 0.5),
            model=model_config,
        )
        .callbacks(TrainingMonitor)
    )
    
    config.multi_agent(
        policies={
            "shared_policy": PolicySpec(
                observation_space=None,
                action_space=None,
                config={},
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"],
    )
    
    return config


def tune_hyperparameters(
    layout_name: str = "cramped_room",
    search_type: str = "medium",
    num_samples: int = 20,
    max_iterations: int = 200,
    scheduler_type: str = "asha",
    local_dir: str = "./results_tuning",
):
    """
    Run hyperparameter optimization.
    
    Args:
        layout_name: Overcooked layout to use
        search_type: Search space size (quick/medium/full)
        num_samples: Number of configurations to try
        max_iterations: Maximum iterations per trial
        scheduler_type: Scheduler (asha/pbt)
        local_dir: Directory to save results
    """
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    register_env("overcooked_mappo", make_overcooked_env)
    
    # Setup search space
    search_space = create_search_space(search_type)
    
    # Setup base environment config
    base_env_config = ENV_CONFIG.copy()
    base_env_config["layout_name"] = layout_name
    
    # Create trainable function
    def trainable(config):
        """Trainable function for Ray Tune."""
        ppo_config = build_config_with_params(config, base_env_config)
        return ppo_config.to_dict()
    
    # Setup scheduler
    if scheduler_type == "asha":
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            max_t=max_iterations,
            grace_period=20,
            reduction_factor=3,
        )
    elif scheduler_type == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=20,
            hyperparam_mutations={
                "lr": tune.loguniform(1e-5, 1e-3),
                "entropy_coeff": tune.loguniform(0.001, 0.1),
            },
        )
    else:
        scheduler = None
    
    # Setup search algorithm (optional)
    search_alg = None
    if search_type in ["medium", "full"]:
        try:
            search_alg = OptunaSearch(
                metric="episode_reward_mean",
                mode="max"
            )
        except ImportError:
            print("Warning: Optuna not available. Using random search.")
    
    print("="*70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Layout: {layout_name}")
    print(f"Search Type: {search_type}")
    print(f"Num Samples: {num_samples}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Search Space: {len(search_space)} parameters")
    print("="*70 + "\n")
    
    # Run tuning
    analysis = tune.run(
        "PPO",
        name=f"tune_{layout_name}",
        config={
            **trainable(search_space),
            **search_space,
        },
        stop={"training_iteration": max_iterations},
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=os.path.abspath(local_dir),
        verbose=1,
        checkpoint_freq=50,
        checkpoint_at_end=True,
    )
    
    # Get best configuration
    best_config = analysis.get_best_config(
        metric="episode_reward_mean",
        mode="max"
    )
    
    best_trial = analysis.get_best_trial(
        metric="episode_reward_mean",
        mode="max"
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest Trial: {best_trial}")
    print(f"Best Reward: {best_trial.last_result.get('episode_reward_mean', 0):.2f}")
    print("\nBest Configuration:")
    print("-"*70)
    for key, value in sorted(best_config.items()):
        if key in search_space:
            print(f"  {key:30s}: {value}")
    print("="*70)
    
    # Save best config
    import json
    config_path = os.path.join(local_dir, "best_config.json")
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    print(f"\n✓ Best config saved to: {config_path}")
    
    # Plot results
    try:
        plot_results(analysis, local_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    ray.shutdown()
    
    return analysis, best_config


def plot_results(analysis, save_dir: str):
    """
    Plot optimization results.
    
    Args:
        analysis: Ray Tune analysis object
        save_dir: Directory to save plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Get all trials
    df = analysis.trial_dataframes
    
    if not df:
        return
    
    # Combine dataframes
    all_data = []
    for trial_id, trial_df in df.items():
        trial_df["trial_id"] = trial_id
        all_data.append(trial_df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Plot reward progression
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward over iterations
    for trial_id in combined_df["trial_id"].unique():
        trial_data = combined_df[combined_df["trial_id"] == trial_id]
        axes[0, 0].plot(
            trial_data["training_iteration"],
            trial_data.get("episode_reward_mean", trial_data.get("env_runners/episode_reward_mean", 0)),
            alpha=0.6
        )
    axes[0, 0].set_xlabel("Training Iteration")
    axes[0, 0].set_ylabel("Episode Reward Mean")
    axes[0, 0].set_title("Reward Progression Across Trials")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Final rewards distribution
    final_rewards = []
    for trial_id in combined_df["trial_id"].unique():
        trial_data = combined_df[combined_df["trial_id"] == trial_id]
        if len(trial_data) > 0:
            final_reward = trial_data.iloc[-1].get(
                "episode_reward_mean",
                trial_data.iloc[-1].get("env_runners/episode_reward_mean", 0)
            )
            final_rewards.append(final_reward)
    
    axes[0, 1].hist(final_rewards, bins=15, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(max(final_rewards), color='red', linestyle='--', label='Best')
    axes[0, 1].set_xlabel("Final Reward")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution of Final Rewards")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning rate vs final reward (if available)
    if "config/lr" in combined_df.columns:
        lr_rewards = []
        for trial_id in combined_df["trial_id"].unique():
            trial_data = combined_df[combined_df["trial_id"] == trial_id]
            if len(trial_data) > 0:
                lr = trial_data.iloc[0]["config/lr"]
                final_reward = trial_data.iloc[-1].get(
                    "episode_reward_mean",
                    trial_data.iloc[-1].get("env_runners/episode_reward_mean", 0)
                )
                lr_rewards.append((lr, final_reward))
        
        if lr_rewards:
            lrs, rewards = zip(*lr_rewards)
            axes[1, 0].scatter(lrs, rewards, alpha=0.6, s=100)
            axes[1, 0].set_xlabel("Learning Rate")
            axes[1, 0].set_ylabel("Final Reward")
            axes[1, 0].set_title("Learning Rate vs Final Reward")
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Entropy coefficient vs final reward (if available)
    if "config/entropy_coeff" in combined_df.columns:
        entropy_rewards = []
        for trial_id in combined_df["trial_id"].unique():
            trial_data = combined_df[combined_df["trial_id"] == trial_id]
            if len(trial_data) > 0:
                entropy = trial_data.iloc[0]["config/entropy_coeff"]
                final_reward = trial_data.iloc[-1].get(
                    "episode_reward_mean",
                    trial_data.iloc[-1].get("env_runners/episode_reward_mean", 0)
                )
                entropy_rewards.append((entropy, final_reward))
        
        if entropy_rewards:
            entropies, rewards = zip(*entropy_rewards)
            axes[1, 1].scatter(entropies, rewards, alpha=0.6, s=100, color='orange')
            axes[1, 1].set_xlabel("Entropy Coefficient")
            axes[1, 1].set_ylabel("Final Reward")
            axes[1, 1].set_title("Entropy Coefficient vs Final Reward")
            axes[1, 1].set_xscale('log')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "tuning_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Results plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for MAPPO")
    
    parser.add_argument("--layout", type=str, default="cramped_room",
                       help="Layout to optimize on")
    parser.add_argument("--search-type", type=str, default="medium",
                       choices=["quick", "medium", "full"],
                       help="Search space size")
    parser.add_argument("--num-samples", type=int, default=20,
                       help="Number of configurations to try")
    parser.add_argument("--max-iterations", type=int, default=200,
                       help="Maximum iterations per trial")
    parser.add_argument("--scheduler", type=str, default="asha",
                       choices=["asha", "pbt", "none"],
                       help="Scheduler type")
    parser.add_argument("--local-dir", type=str, default="./results_tuning",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    tune_hyperparameters(
        layout_name=args.layout,
        search_type=args.search_type,
        num_samples=args.num_samples,
        max_iterations=args.max_iterations,
        scheduler_type=args.scheduler,
        local_dir=args.local_dir,
    )


if __name__ == "__main__":
    main()