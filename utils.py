# utils.py
"""
Utility functions for MAPPO Overcooked training
- Visualization
- Analysis
- Debugging
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add it to the system path
if src_path not in sys.path:
    sys.path.append(src_path)

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.actions import Action


def plot_training_curves(result_dir: str, metrics: List[str] = None, save_path: Optional[str] = None):
    """
    Plot training curves from Ray results.
    
    Args:
        result_dir: Path to Ray results directory
        metrics: List of metrics to plot (default: reward, loss)
        save_path: Path to save plot (optional)
    """
    if metrics is None:
        metrics = [
            "episode_reward_mean",
            "policy_loss",
            "vf_loss",
            "entropy",
        ]
    
    result_path = Path(result_dir)
    
    # Find progress.csv
    progress_files = list(result_path.rglob("progress.csv"))
    if not progress_files:
        print(f"No progress.csv found in {result_dir}")
        return
    
    import pandas as pd
    df = pd.read_csv(progress_files[0])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics[:4]):  # Max 4 plots
        if metric in df.columns:
            axes[i].plot(df["training_iteration"], df[metric])
            axes[i].set_xlabel("Training Iteration")
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{metric}")
            axes[i].grid(True)
        else:
            print(f"Metric {metric} not found in progress.csv")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def visualize_episode(
    checkpoint_path: str,
    layout_name: str = "cramped_room",
    save_path: Optional[str] = None,
):
    """
    Visualize a single episode with trained agents.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        layout_name: Layout to visualize
        save_path: Path to save visualization (optional)
    """
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
    from ray.tune.registry import register_env
    from config import ENV_CONFIG
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    register_env("overcooked_mappo", make_overcooked_env)
    
    # Load agent
    agent = Algorithm.from_checkpoint(checkpoint_path)
    
    # Create environment
    env_config = ENV_CONFIG.copy()
    env_config["layout_name"] = layout_name
    env = OvercookedMAPPOEnv(env_config)
    
    # Run episode
    obs = env.reset()
    done = False
    states = [env.env.state]
    actions_taken = []
    
    while not done:
        # Get actions
        actions = {}
        for agent_id in ["agent_0", "agent_1"]:
            action = agent.compute_single_action(
                obs[agent_id],
                policy_id="shared_policy"
            )
            actions[agent_id] = action
        
        # Step
        obs, rewards, dones, infos = env.step(actions)
        done = dones["__all__"]
        
        states.append(env.env.state)
        actions_taken.append(actions)
    
    ray.shutdown()
    
    # Visualize trajectory
    print(f"Episode completed: {len(states)} steps")
    
    # Can use StateVisualizer for rendering (requires additional setup)
    # For now, print summary
    print_episode_summary(states, actions_taken)
    
    return states, actions_taken


def print_episode_summary(states: List[OvercookedState], actions: List[Dict]):
    """Print a summary of an episode."""
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    
    print(f"Total steps: {len(states)}")
    print(f"Total deliveries: {states[-1].order_list if hasattr(states[-1], 'order_list') else 'N/A'}")
    
    # Action distribution
    action_names = [a.name for a in Action.ALL_ACTIONS]
    action_counts_0 = {name: 0 for name in action_names}
    action_counts_1 = {name: 0 for name in action_names}
    
    for action_dict in actions:
        action_0 = Action.ALL_ACTIONS[action_dict["agent_0"]]
        action_1 = Action.ALL_ACTIONS[action_dict["agent_1"]]
        action_counts_0[action_0.name] += 1
        action_counts_1[action_1.name] += 1
    
    print("\nAgent 0 Action Distribution:")
    for name, count in action_counts_0.items():
        pct = 100 * count / len(actions) if actions else 0
        print(f"  {name:12s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nAgent 1 Action Distribution:")
    for name, count in action_counts_1.items():
        pct = 100 * count / len(actions) if actions else 0
        print(f"  {name:12s}: {count:4d} ({pct:5.1f}%)")
    
    print("="*60)


def analyze_checkpoint_performance(
    checkpoint_path: str,
    layouts: List[str] = None,
    num_episodes: int = 10,
):
    """
    Analyze checkpoint performance across multiple layouts.
    
    Args:
        checkpoint_path: Path to checkpoint
        layouts: List of layouts to test
        num_episodes: Episodes per layout
        
    Returns:
        Dictionary with performance metrics
    """
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
    from ray.tune.registry import register_env
    from config import ENV_CONFIG
    
    if layouts is None:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    register_env("overcooked_mappo", make_overcooked_env)
    
    # Load agent
    agent = Algorithm.from_checkpoint(checkpoint_path)
    
    results = {}
    
    for layout in layouts:
        print(f"\nTesting on {layout}...")
        
        env_config = ENV_CONFIG.copy()
        env_config["layout_name"] = layout
        env = OvercookedMAPPOEnv(env_config)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                actions = {}
                for agent_id in ["agent_0", "agent_1"]:
                    action = agent.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy"
                    )
                    actions[agent_id] = action
                
                obs, rewards, dones, infos = env.step(actions)
                done = dones["__all__"]
                episode_reward += rewards["agent_0"]
                step_count += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
        
        results[layout] = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }
        
        print(f"  Mean reward: {results[layout]['mean_reward']:.2f} ± {results[layout]['std_reward']:.2f}")
    
    ray.shutdown()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    for layout, metrics in results.items():
        print(f"\n{layout}:")
        print(f"  Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print(f"  Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    
    return results


def compare_checkpoints(
    checkpoint_paths: Dict[str, str],
    layout_name: str = "cramped_room",
    num_episodes: int = 10,
):
    """
    Compare multiple checkpoints.
    
    Args:
        checkpoint_paths: Dict mapping names to checkpoint paths
        layout_name: Layout to test on
        num_episodes: Number of episodes
        
    Returns:
        Comparison results
    """
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
    from ray.tune.registry import register_env
    from config import ENV_CONFIG
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    register_env("overcooked_mappo", make_overcooked_env)
    
    results = {}
    
    for name, checkpoint_path in checkpoint_paths.items():
        print(f"\nEvaluating {name}...")
        
        agent = Algorithm.from_checkpoint(checkpoint_path)
        
        env_config = ENV_CONFIG.copy()
        env_config["layout_name"] = layout_name
        env = OvercookedMAPPOEnv(env_config)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                actions = {}
                for agent_id in ["agent_0", "agent_1"]:
                    action = agent.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy"
                    )
                    actions[agent_id] = action
                
                obs, rewards, dones, infos = env.step(actions)
                done = dones["__all__"]
                episode_reward += rewards["agent_0"]
            
            episode_rewards.append(episode_reward)
        
        results[name] = {
            "mean": np.mean(episode_rewards),
            "std": np.std(episode_rewards),
            "rewards": episode_rewards,
        }
    
    ray.shutdown()
    
    # Print comparison
    print("\n" + "="*60)
    print(f"CHECKPOINT COMPARISON ({layout_name})")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:20s}: {metrics['mean']:7.2f} ± {metrics['std']:5.2f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, metrics in results.items():
        ax.plot(metrics['rewards'], label=name, marker='o', alpha=0.7)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Checkpoint Comparison - {layout_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_training_curves(result_dir)")
    print("  - visualize_episode(checkpoint_path, layout_name)")
    print("  - analyze_checkpoint_performance(checkpoint_path, layouts)")
    print("  - compare_checkpoints(checkpoint_paths, layout_name)")