# train_mappo.py
"""
MAPPO Training Script for Overcooked-AI
Trains a shared policy across both agents using PPO with centralized value function
Updated for Ray RLlib 2.30+ / 3.0 API & Gymnasium
"""

# --- SUPPRESS WARNINGS ---
import warnings
import os
# Filter noisy Ray/Gym deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# -------------------------

import argparse
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Import gymnasium instead of gym
import gymnasium as gym

# Setup paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
from config import ENV_CONFIG, TRAINING_CONFIG, EXPERIMENT_CONFIG

def setup_mappo_config(env_config, training_config):
    """
    Configure PPO to work as MAPPO (Multi-Agent PPO).
    """
    
    # Create PPO config
    config = (
        PPOConfig()
        # --- CRITICAL FIX FOR RAY 2.30+ ---
        # Disable the new API stack to support legacy model configs and callbacks
        .api_stack(
            enable_rl_module_and_learner=False, 
            enable_env_runner_and_connector_v2=False
        )
        # ----------------------------------
        .environment(
            env="overcooked_mappo",
            env_config=env_config,
        )
        .framework(training_config["framework"])
        .resources(
            num_gpus=training_config["num_gpus"],
        )
        .env_runners(
            num_env_runners=training_config["num_workers"],
            num_envs_per_env_runner=training_config["num_envs_per_worker"],
            # Auto-calculate fragment length to match batch size
            rollout_fragment_length="auto",
            batch_mode=training_config["batch_mode"],
            num_cpus_per_env_runner=training_config["num_cpus_per_worker"],
        )
        .training(
            train_batch_size=training_config["train_batch_size"],
            minibatch_size=training_config["sgd_minibatch_size"],
            num_sgd_iter=training_config["num_sgd_iter"],
            lr=training_config["lr"],
            gamma=training_config["gamma"],
            lambda_=training_config["lambda"],
            clip_param=training_config["clip_param"],
            vf_clip_param=training_config["vf_clip_param"],
            entropy_coeff=training_config["entropy_coeff"],
            vf_loss_coeff=training_config["vf_loss_coeff"],
            model=training_config["model"],
        )
        .evaluation(
            evaluation_interval=training_config["evaluation_interval"],
            evaluation_duration=training_config["evaluation_duration"],
            evaluation_duration_unit=training_config["evaluation_duration_unit"],
            evaluation_num_env_runners=training_config["evaluation_num_workers"],
        )
        .debugging(
            log_level=training_config["log_level"],
        )
    )
    
    # ===== MAPPO CONFIGURATION =====
    # Define a shared policy for both agents (parameter sharing)
    config.multi_agent(
        policies={
            "shared_policy": PolicySpec(
                observation_space=None,  # Will be auto-detected
                action_space=None,       # Will be auto-detected
                config={},
            ),
        },
        # Map both agents to the same policy
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        # Only train this one policy
        policies_to_train=["shared_policy"],
    )
    
    return config


def train_mappo(
    layout_name="cramped_room",
    num_iterations=500,
    checkpoint_freq=10,
    use_phi=False,
    num_workers=4,
    local_dir="./results",
):
    """
    Main training loop for MAPPO on Overcooked.
    """
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("overcooked_mappo", make_overcooked_env)
    
    # Update configs
    env_config = ENV_CONFIG.copy()
    env_config["layout_name"] = layout_name
    env_config["use_phi"] = use_phi
    
    training_config = TRAINING_CONFIG.copy()
    training_config["num_workers"] = num_workers
    
    # Setup MAPPO config
    config = setup_mappo_config(env_config, training_config)
    
    # Configure experiment
    stop_criteria = {
        "training_iteration": num_iterations,
    }
    
    # Ensure storage path is absolute to prevent path resolution errors
    storage_path = os.path.abspath(local_dir)

    # Run training
    print("=" * 60)
    print(f"Starting MAPPO training on {layout_name}")
    print(f"Configuration:")
    print(f"  - Layout: {layout_name}")
    print(f"  - Use Phi (shaped rewards): {use_phi}")
    print(f"  - Env Runners (Workers): {num_workers}")
    print(f"  - Training iterations: {num_iterations}")
    print(f"  - Storage Path: {storage_path}")
    print("=" * 60)
    
    results = tune.run(
        "PPO",
        name=f"mappo_{layout_name}",
        config=config.to_dict(),
        stop=stop_criteria,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        storage_path=storage_path,
        verbose=1,
    )
    
    # --- ROBUST CHECKPOINT RETRIEVAL ---
    best_checkpoint = None
    try:
        # Try new API stack metric location first
        best_trial = results.get_best_trial("env_runners/episode_reward_mean", mode="max")
        metric_key = "env_runners/episode_reward_mean"
        
        if not best_trial:
            # Fallback to old metric location
            best_trial = results.get_best_trial("episode_reward_mean", mode="max")
            metric_key = "episode_reward_mean"
            
        if best_trial:
            best_checkpoint = results.get_best_checkpoint(
                trial=best_trial,
                metric=metric_key,
                mode="max"
            )
            print("\n" + "=" * 60)
            print("Training completed!")
            print(f"Best checkpoint: {best_checkpoint}")
            print("=" * 60)
        else:
            print("\nWarning: Could not determine best trial from metrics.")
            
    except Exception as e:
        print(f"\nWarning: Error retrieving best checkpoint ({e}).")
        print("Check the results directory manually for checkpoints.")
    
    ray.shutdown()
    
    return results, best_checkpoint


def evaluate_checkpoint(checkpoint_path, layout_name="cramped_room", num_episodes=10):
    """
    Evaluate a trained MAPPO checkpoint.
    """
    from ray.rllib.algorithms.algorithm import Algorithm
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    register_env("overcooked_mappo", make_overcooked_env)
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Load trained agent
    # We suppress logs here to minimize the TensorBoard thread error at exit
    agent = Algorithm.from_checkpoint(checkpoint_path)
    
    # Create environment
    env_config = ENV_CONFIG.copy()
    env_config["layout_name"] = layout_name
    env = OvercookedMAPPOEnv(env_config)
    
    print(f"\nEvaluating on {layout_name} for {num_episodes} episodes...")
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            actions = {}
            for agent_id in ["agent_0", "agent_1"]:
                # Use compute_single_action (legacy API, but correct for this stack)
                action = agent.compute_single_action(
                    obs[agent_id],
                    policy_id="shared_policy"
                )
                actions[agent_id] = action
            
            step_result = env.step(actions)
            
            if len(step_result) == 5:
                obs, rewards, terminated, truncated, infos = step_result
                done = terminated["__all__"] or truncated["__all__"]
            else:
                obs, rewards, done, infos = step_result
                if isinstance(done, dict):
                     done = done["__all__"]
            
            episode_reward += rewards["agent_0"]
            step_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nMean reward over {num_episodes} episodes: {mean_reward:.2f}")
    
    ray.shutdown()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO on Overcooked")
    
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-phi", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--local-dir", type=str, default="./results")
    parser.add_argument("--eval-checkpoint", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.eval_checkpoint:
        evaluate_checkpoint(
            args.eval_checkpoint,
            layout_name=args.layout,
            num_episodes=args.eval_episodes
        )
    else:
        results, best_checkpoint = train_mappo(
            layout_name=args.layout,
            num_iterations=args.iterations,
            checkpoint_freq=args.checkpoint_freq,
            use_phi=args.use_phi,
            num_workers=args.workers,
            local_dir=args.local_dir,
        )
        
        if best_checkpoint:
            print(f"\nTo evaluate this checkpoint, run:")
            print(f"python train_mappo.py --eval-checkpoint {best_checkpoint} --layout {args.layout}")

if __name__ == "__main__":
    main()