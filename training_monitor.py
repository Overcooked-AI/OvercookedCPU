# training_monitor.py
"""
Real-time training monitoring with TensorBoard integration
Tracks training progress, coordination metrics, and generates visualizations
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from collections import deque
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress specific Gym warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

# FIX: Conditional import for Ray 2.30+ compatibility
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode, RolloutWorker

from coordination_metrics import CoordinationMetricsTracker


class TrainingMonitor(DefaultCallbacks):
    """
    RLlib callback for monitoring training with coordination metrics.
    Logs to TensorBoard and tracks detailed episode statistics.
    """
    
    def __init__(self):
        super().__init__()
        self.episode_metrics = {}
        self.writer = None
        self.log_dir = None
        
        # Rolling statistics
        self.recent_rewards = deque(maxlen=100)
        self.recent_soup_counts = deque(maxlen=100)
        self.recent_collisions = deque(maxlen=100)
    
    def on_algorithm_init(self, *, algorithm, **kwargs):
        """Initialize monitoring when training starts."""
        # Setup TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.log_dir = Path(algorithm.logdir) / "tensorboard"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            print(f"📊 TensorBoard logging to: {self.log_dir}")
            print(f"   View with: tensorboard --logdir {self.log_dir}")
        
        # Create metrics tracker for each worker
        # FIX: Robustly handle algorithm.workers access for different Ray versions
        worker_group = None
        
        # 1. Try env_runner_group (Newest Ray)
        if hasattr(algorithm, "env_runner_group") and algorithm.env_runner_group is not None:
            worker_group = algorithm.env_runner_group
            
        # 2. Try workers (Older Ray) - safely check if it exists and is not None
        if worker_group is None and hasattr(algorithm, "workers"):
            try:
                # Some Ray versions treat .workers as a property, others as a method
                # We check if it is callable without triggering property access that might fail
                workers_attr = algorithm.workers
                if callable(workers_attr):
                    worker_group = workers_attr()
                else:
                    worker_group = workers_attr
            except Exception:
                # If accessing the property fails, we just move on
                pass

        # Apply tracker to workers
        if worker_group is not None:
            if hasattr(worker_group, "foreach_worker"):
                worker_group.foreach_worker(
                    lambda w: setattr(w, "coordination_tracker", CoordinationMetricsTracker())
                )
            elif hasattr(worker_group, "foreach_env_runner"):
                 worker_group.foreach_env_runner(
                    lambda w: setattr(w, "coordination_tracker", CoordinationMetricsTracker())
                )
    
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the start of each episode."""
        # Reset coordination tracker
        if hasattr(worker, "coordination_tracker"):
            worker.coordination_tracker.reset()
        
        # Initialize episode metrics storage
        episode.user_data["coordination_metrics"] = {}
        episode.user_data["step_rewards"] = []
    
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """Called at each step of the episode."""
        # Extract state and actions from the episode
        if hasattr(worker, "coordination_tracker"):
            # Get the last step's info - checking for modern Ray API methods
            # Use last_info_for() if available (legacy/current hybrid)
            infos = None
            if hasattr(episode, 'last_info_for'):
                infos = episode.last_info_for()
            
            if infos and "agent_0" in infos:
                # Get actions from episode
                actions = {}
                rewards = {}
                for agent_id in ["agent_0", "agent_1"]:
                    if hasattr(episode, 'last_action_for') and agent_id in episode.last_action_for():
                        actions[agent_id] = episode.last_action_for(agent_id)
                    if hasattr(episode, 'last_reward_for') and agent_id in episode.last_reward_for():
                        rewards[agent_id] = episode.last_reward_for(agent_id)
                
                # Update coordination tracker (we'll approximate state from info)
                # Note: For full state tracking, you'd need to modify the env
                # to return state in info dict
                try:
                    worker.coordination_tracker.update(
                        state=None,  # Would need state from env
                        actions=actions,
                        rewards=rewards,
                        info=infos["agent_0"]
                    )
                except Exception as e:
                    pass  # Skip if state not available
            
            # Track step rewards
            if hasattr(episode, 'last_reward_for'):
                episode.user_data["step_rewards"].append(
                    episode.last_reward_for("agent_0")
                )
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Called when episode ends - compute and log coordination metrics."""
        
        # Get coordination metrics
        if hasattr(worker, "coordination_tracker"):
            coord_metrics = worker.coordination_tracker.get_metrics()
            episode.user_data["coordination_metrics"] = coord_metrics
            
            # Log key coordination metrics
            for key, value in coord_metrics.items():
                episode.custom_metrics[f"coord/{key}"] = value
        
        # Compute episode statistics
        # Handle different Ray versions for episode stats
        if hasattr(episode, 'total_reward'):
            episode_reward = episode.total_reward
            episode_length = episode.length
        else:
            # Fallback for newer API or if total_reward isn't directly available
            episode_reward = sum(episode.user_data.get("step_rewards", [0]))
            episode_length = len(episode.user_data.get("step_rewards", []))
        
        # Store in custom metrics for aggregation
        episode.custom_metrics["episode_reward"] = episode_reward
        episode.custom_metrics["episode_length"] = episode_length
        
        # Track rolling statistics
        self.recent_rewards.append(episode_reward)
        
        soups = coord_metrics.get("soups_delivered", 0) if coord_metrics else 0
        self.recent_soup_counts.append(soups)
        
        collisions = coord_metrics.get("collisions_per_100_steps", 0) if coord_metrics else 0
        self.recent_collisions.append(collisions)
        
        # Add rolling averages to metrics
        if len(self.recent_rewards) > 0:
            episode.custom_metrics["rolling_avg_reward"] = np.mean(self.recent_rewards)
            episode.custom_metrics["rolling_avg_soups"] = np.mean(self.recent_soup_counts)
            episode.custom_metrics["rolling_avg_collisions"] = np.mean(self.recent_collisions)
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called after each training iteration - log to TensorBoard."""
        
        if not TENSORBOARD_AVAILABLE or self.writer is None:
            return
        
        iteration = result.get("training_iteration", 0)
        timesteps = result.get("timesteps_total", 0)
        
        # === Core Training Metrics ===
        self._log_scalar("train/episode_reward_mean", 
                        result.get("env_runners", {}).get("episode_reward_mean", 
                        result.get("episode_reward_mean", 0)), iteration)
        
        self._log_scalar("train/episode_len_mean",
                        result.get("env_runners", {}).get("episode_len_mean",
                        result.get("episode_len_mean", 0)), iteration)
        
        # Policy metrics
        if "info" in result and "learner" in result["info"]:
            learner_info = result["info"]["learner"].get("shared_policy", {})
            if not learner_info and "default_policy" in result["info"]["learner"]:
                 learner_info = result["info"]["learner"]["default_policy"]

            learner_stats = learner_info.get("learner_stats", {})
            
            self._log_scalar("policy/loss", 
                           learner_stats.get("total_loss", 0), 
                           iteration)
            self._log_scalar("policy/policy_loss",
                           learner_stats.get("policy_loss", 0),
                           iteration)
            self._log_scalar("policy/vf_loss",
                           learner_stats.get("vf_loss", 0),
                           iteration)
            self._log_scalar("policy/entropy",
                           learner_stats.get("entropy", 0),
                           iteration)
            self._log_scalar("policy/kl_divergence",
                           learner_stats.get("kl", 0),
                           iteration)
        
        # === Coordination Metrics ===
        custom_metrics = result.get("custom_metrics", {})
        
        for key, value in custom_metrics.items():
            if key.startswith("coord/"):
                metric_name = key.replace("coord/", "")
                self._log_scalar(f"coordination/{metric_name}", value, iteration)
        
        # Rolling averages
        if "rolling_avg_reward" in custom_metrics:
            self._log_scalar("train/rolling_avg_reward",
                           custom_metrics["rolling_avg_reward"], iteration)
        if "rolling_avg_soups" in custom_metrics:
            self._log_scalar("coordination/rolling_avg_soups",
                           custom_metrics["rolling_avg_soups"], iteration)
        if "rolling_avg_collisions" in custom_metrics:
            self._log_scalar("coordination/rolling_avg_collisions",
                           custom_metrics["rolling_avg_collisions"], iteration)
        
        # === Resource Metrics ===
        self._log_scalar("resources/timesteps_total", timesteps, iteration)
        
        timers = result.get("timers", {})
        if "sample_time_ms" in timers:
            self._log_scalar("performance/sample_time_ms",
                           timers["sample_time_ms"], iteration)
        if "learn_time_ms" in timers:
            self._log_scalar("performance/learn_time_ms",
                           timers["learn_time_ms"], iteration)
        
        # Flush to disk
        self.writer.flush()
        
        # Print progress
        self._print_progress(iteration, result, custom_metrics)
    
    def _log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        if self.writer and value is not None:
            self.writer.add_scalar(tag, value, step)
    
    def _print_progress(self, iteration: int, result: Dict, custom_metrics: Dict):
        """Print training progress to console."""
        
        reward = result.get("env_runners", {}).get("episode_reward_mean",
                  result.get("episode_reward_mean", 0))
        length = result.get("env_runners", {}).get("episode_len_mean",
                  result.get("episode_len_mean", 0))
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        print(f"  Reward: {reward:.2f} | Episode Length: {length:.1f}")
        
        if "coord/soups_delivered" in custom_metrics:
            soups = custom_metrics["coord/soups_delivered"]
            steps_per_soup = custom_metrics.get("coord/steps_per_soup", 0)
            print(f"  Soups: {soups:.1f} | Steps/Soup: {steps_per_soup:.1f}")
        
        if "coord/collisions_per_100_steps" in custom_metrics:
            collisions = custom_metrics["coord/collisions_per_100_steps"]
            print(f"  Collisions: {collisions:.2f} per 100 steps")
        
        if "rolling_avg_reward" in custom_metrics:
            rolling_reward = custom_metrics["rolling_avg_reward"]
            print(f"  Rolling Avg Reward (100 eps): {rolling_reward:.2f}")
    
    def on_algorithm_end(self, *, algorithm, **kwargs):
        """Called when training ends."""
        if self.writer:
            try:
                self.writer.close()
            except Exception:
                pass
            print(f"\n✅ TensorBoard logs saved to: {self.log_dir}")


class LiveMonitor:
    """
    Standalone monitor for tracking training progress in real-time.
    Reads from Ray results directory and displays live updates.
    """
    
    def __init__(self, results_dir: str, refresh_interval: float = 5.0):
        """
        Args:
            results_dir: Path to Ray results directory
            refresh_interval: How often to update (seconds)
        """
        self.results_dir = Path(results_dir)
        self.refresh_interval = refresh_interval
        self.last_iteration = 0
    
    def monitor(self):
        """Start monitoring training progress."""
        print(f"📊 Monitoring training from: {self.results_dir}")
        print(f"Refresh interval: {self.refresh_interval}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self._update_display()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
    
    def _update_display(self):
        """Update the display with latest training stats."""
        # Find progress.csv
        progress_files = list(self.results_dir.rglob("progress.csv"))
        
        if not progress_files:
            print("Waiting for training to start...")
            return
        
        import pandas as pd
        try:
            df = pd.read_csv(progress_files[0])
            
            if len(df) == 0:
                return
            
            latest = df.iloc[-1]
            iteration = int(latest.get("training_iteration", 0))
            
            # Only print if new iteration
            if iteration <= self.last_iteration:
                return
            
            self.last_iteration = iteration
            
            # Clear screen (optional, comment out if annoying)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("="*70)
            print(f"LIVE TRAINING MONITOR - Iteration {iteration}")
            print("="*70)
            
            # Core metrics
            reward = latest.get("episode_reward_mean", 0)
            length = latest.get("episode_len_mean", 0)
            timesteps = latest.get("timesteps_total", 0)
            
            print(f"\n📊 Training Progress:")
            print(f"  Timesteps: {timesteps:,}")
            print(f"  Episode Reward: {reward:.2f}")
            print(f"  Episode Length: {length:.1f}")
            
            # Coordination metrics (if available)
            coord_cols = [col for col in df.columns if "coord/" in col]
            if coord_cols:
                print(f"\n🤝 Coordination Metrics:")
                for col in coord_cols[:5]:  # Show top 5
                    metric_name = col.replace("coord/", "")
                    value = latest.get(col, 0)
                    print(f"  {metric_name}: {value:.2f}")
            
            # Learning metrics
            if "info/learner/shared_policy/learner_stats/entropy" in df.columns:
                entropy = latest.get("info/learner/shared_policy/learner_stats/entropy", 0)
                print(f"\n🧠 Policy Metrics:")
                print(f"  Entropy: {entropy:.4f}")
            
            print("\n" + "="*70)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor MAPPO training")
    parser.add_argument("--results-dir", type=str, default="./results",
                       help="Path to Ray results directory")
    parser.add_argument("--refresh", type=float, default=5.0,
                       help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor = LiveMonitor(args.results_dir, args.refresh)
    monitor.monitor()