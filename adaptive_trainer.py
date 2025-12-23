# adaptive_trainer.py
"""
Adaptive MAPPO training with curriculum learning and dynamic learning rate
Automatically adjusts training parameters based on performance
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
from config import ENV_CONFIG, TRAINING_CONFIG
from training_monitor import TrainingMonitor


class CurriculumSchedule:
    """
    Defines a curriculum learning schedule for Overcooked.
    Progressively increases task difficulty.
    """
    
    def __init__(self, schedule_type: str = "progressive"):
        """
        Args:
            schedule_type: 
                - "progressive": Start easy, increase difficulty
                - "mixed": Mix of difficulties
                - "reverse": Start hard, decrease difficulty (for robust learning)
        """
        self.schedule_type = schedule_type
        
        # Define layout difficulty ranking (subjective, based on coordination needs)
        self.layouts_by_difficulty = {
            "easy": ["cramped_room"],
            "medium": ["asymmetric_advantages", "coordination_ring"],
            "hard": ["forced_coordination", "counter_circuit"],
        }
        
        # All layouts in order
        self.progressive_order = (
            self.layouts_by_difficulty["easy"] +
            self.layouts_by_difficulty["medium"] +
            self.layouts_by_difficulty["hard"]
        )
    
    def get_layout_for_iteration(self, iteration: int, total_iterations: int) -> str:
        """
        Determine which layout to train on based on current iteration.
        
        Args:
            iteration: Current training iteration
            total_iterations: Total planned iterations
            
        Returns:
            Layout name to use
        """
        progress = iteration / total_iterations
        
        if self.schedule_type == "progressive":
            # Start with easy, gradually introduce harder layouts
            if progress < 0.3:
                # First 30%: easy layouts only
                return np.random.choice(self.layouts_by_difficulty["easy"])
            elif progress < 0.7:
                # Next 40%: easy + medium
                return np.random.choice(
                    self.layouts_by_difficulty["easy"] + 
                    self.layouts_by_difficulty["medium"]
                )
            else:
                # Final 30%: all layouts
                return np.random.choice(self.progressive_order)
        
        elif self.schedule_type == "mixed":
            # Always mix all difficulties
            return np.random.choice(self.progressive_order)
        
        elif self.schedule_type == "reverse":
            # Start hard, then easier (for robustness)
            if progress < 0.5:
                return np.random.choice(self.layouts_by_difficulty["hard"])
            else:
                return np.random.choice(self.progressive_order)
        
        return "cramped_room"  # Fallback


class AdaptiveLearningRate:
    """
    Dynamically adjusts learning rate based on training progress.
    """
    
    def __init__(
        self,
        initial_lr: float = 5e-4,
        min_lr: float = 1e-5,
        adaptation_strategy: str = "cosine_annealing"
    ):
        """
        Args:
            initial_lr: Starting learning rate
            min_lr: Minimum learning rate
            adaptation_strategy:
                - "cosine_annealing": Smooth decay following cosine curve
                - "step_decay": Decay at fixed intervals
                - "plateau": Reduce when performance plateaus
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.strategy = adaptation_strategy
        self.performance_history = []
    
    def get_lr(self, iteration: int, total_iterations: int, performance: Optional[float] = None) -> float:
        """
        Compute learning rate for current iteration.
        
        Args:
            iteration: Current iteration
            total_iterations: Total iterations planned
            performance: Recent performance metric (optional, for plateau strategy)
            
        Returns:
            Learning rate to use
        """
        progress = iteration / total_iterations
        
        if self.strategy == "cosine_annealing":
            # Cosine annealing: smooth decay from initial_lr to min_lr
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
            return lr
        
        elif self.strategy == "step_decay":
            # Step decay: reduce by factor of 0.5 every 25% of training
            decay_factor = 0.5 ** int(progress / 0.25)
            lr = max(self.initial_lr * decay_factor, self.min_lr)
            return lr
        
        elif self.strategy == "plateau":
            # Reduce when performance plateaus
            if performance is not None:
                self.performance_history.append(performance)
                
                # Check for plateau (last 20 iterations have < 5% improvement)
                if len(self.performance_history) >= 20:
                    recent = self.performance_history[-20:]
                    improvement = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-8)
                    
                    if improvement < 0.05:
                        # Plateau detected, reduce LR
                        return max(self.initial_lr * 0.5 * (1 - progress), self.min_lr)
            
            # Default: linear decay
            lr = self.initial_lr * (1 - progress * 0.9) + self.min_lr * progress
            return max(lr, self.min_lr)
        
        return self.initial_lr


class AdaptiveTrainer:
    """
    Main adaptive training orchestrator with curriculum learning and dynamic LR.
    """
    
    def __init__(
        self,
        initial_layout: str = "cramped_room",
        num_iterations: int = 1000,
        enable_curriculum: bool = True,
        curriculum_type: str = "progressive",
        enable_adaptive_lr: bool = True,
        lr_strategy: str = "cosine_annealing",
        initial_lr: float = 5e-4,
        checkpoint_freq: int = 20,
        local_dir: str = "./results_adaptive"
    ):
        self.initial_layout = initial_layout
        self.num_iterations = num_iterations
        self.enable_curriculum = enable_curriculum
        self.enable_adaptive_lr = enable_adaptive_lr
        self.checkpoint_freq = checkpoint_freq
        self.local_dir = local_dir
        
        # Setup curriculum
        self.curriculum = CurriculumSchedule(curriculum_type) if enable_curriculum else None
        
        # Setup adaptive LR
        self.lr_scheduler = AdaptiveLearningRate(
            initial_lr=initial_lr,
            adaptation_strategy=lr_strategy
        ) if enable_adaptive_lr else None
        
        self.current_lr = initial_lr
        self.current_layout = initial_layout
    
    def train(self, num_workers: int = 4, use_phi: bool = False):
        """
        Run adaptive training with curriculum and/or dynamic LR.
        
        Args:
            num_workers: Number of parallel workers
            use_phi: Whether to use reward shaping
        """
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        register_env("overcooked_mappo", make_overcooked_env)
        
        print("="*70)
        print("ADAPTIVE MAPPO TRAINING")
        print("="*70)
        print(f"Curriculum Learning: {'✓' if self.enable_curriculum else '✗'}")
        print(f"Adaptive Learning Rate: {'✓' if self.enable_adaptive_lr else '✗'}")
        print(f"Total Iterations: {self.num_iterations}")
        print(f"Workers: {num_workers}")
        print("="*70 + "\n")
        
        # Build base config
        env_config = ENV_CONFIG.copy()
        env_config["layout_name"] = self.current_layout
        env_config["use_phi"] = use_phi
        
        training_config = TRAINING_CONFIG.copy()
        training_config["num_workers"] = num_workers
        training_config["lr"] = self.current_lr
        
        # Create PPO config
        config = self._build_config(env_config, training_config)
        
        # Create algorithm instance for manual training loop
        from ray.rllib.algorithms.ppo import PPO
        algorithm = PPO(config=config)
        
        # Training loop with adaptive adjustments
        best_reward = float('-inf')
        
        for iteration in range(1, self.num_iterations + 1):
            # Update curriculum (change layout)
            if self.enable_curriculum and self.curriculum:
                new_layout = self.curriculum.get_layout_for_iteration(
                    iteration, self.num_iterations
                )
                if new_layout != self.current_layout:
                    print(f"\n📚 Curriculum: Switching to layout '{new_layout}'")
                    self.current_layout = new_layout
                    
                    # Update environment config
                    new_env_config = env_config.copy()
                    new_env_config["layout_name"] = new_layout
                    
                    # Recreate workers with new layout
                    algorithm.workers.foreach_worker(
                        lambda w: w.foreach_env(
                            lambda e: setattr(e, 'layout_name', new_layout)
                        )
                    )
            
            # Update learning rate
            if self.enable_adaptive_lr and self.lr_scheduler:
                new_lr = self.lr_scheduler.get_lr(
                    iteration, self.num_iterations,
                    performance=best_reward
                )
                
                if abs(new_lr - self.current_lr) > 1e-7:
                    print(f"🎓 Adaptive LR: {self.current_lr:.6f} → {new_lr:.6f}")
                    self.current_lr = new_lr
                    
                    # Update algorithm's learning rate
                    algorithm.get_policy("shared_policy").config["lr"] = new_lr
            
            # Train one iteration
            result = algorithm.train()
            
            # Extract metrics
            reward = result.get("env_runners", {}).get("episode_reward_mean",
                     result.get("episode_reward_mean", 0))
            
            if reward > best_reward:
                best_reward = reward
            
            # Print progress
            if iteration % 10 == 0 or iteration == 1:
                self._print_progress(iteration, result, self.current_layout, self.current_lr)
            
            # Save checkpoint
            if iteration % self.checkpoint_freq == 0:
                checkpoint_path = algorithm.save(checkpoint_dir=self.local_dir)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_checkpoint = algorithm.save(checkpoint_dir=self.local_dir)
        print(f"\n✅ Training complete! Final checkpoint: {final_checkpoint}")
        
        algorithm.stop()
        ray.shutdown()
        
        return final_checkpoint
    
    def _build_config(self, env_config: Dict, training_config: Dict) -> PPOConfig:
        """Build PPO configuration."""
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
            .framework(training_config["framework"])
            .resources(num_gpus=training_config["num_gpus"])
            .env_runners(
                num_env_runners=training_config["num_workers"],
                num_envs_per_env_runner=training_config["num_envs_per_worker"],
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
    
    def _print_progress(self, iteration: int, result: Dict, layout: str, lr: float):
        """Print training progress."""
        reward = result.get("env_runners", {}).get("episode_reward_mean",
                 result.get("episode_reward_mean", 0))
        length = result.get("env_runners", {}).get("episode_len_mean",
                 result.get("episode_len_mean", 0))
        
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{self.num_iterations}")
        print(f"Layout: {layout} | LR: {lr:.6f}")
        print(f"Reward: {reward:.2f} | Length: {length:.1f}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive MAPPO Training")
    
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-phi", action="store_true")
    
    # Curriculum options
    parser.add_argument("--enable-curriculum", action="store_true",
                       help="Enable curriculum learning")
    parser.add_argument("--curriculum-type", type=str, default="progressive",
                       choices=["progressive", "mixed", "reverse"])
    
    # Adaptive LR options
    parser.add_argument("--enable-adaptive-lr", action="store_true",
                       help="Enable adaptive learning rate")
    parser.add_argument("--lr-strategy", type=str, default="cosine_annealing",
                       choices=["cosine_annealing", "step_decay", "plateau"])
    parser.add_argument("--initial-lr", type=float, default=5e-4)
    
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--local-dir", type=str, default="./results_adaptive")
    
    args = parser.parse_args()
    
    trainer = AdaptiveTrainer(
        initial_layout=args.layout,
        num_iterations=args.iterations,
        enable_curriculum=args.enable_curriculum,
        curriculum_type=args.curriculum_type,
        enable_adaptive_lr=args.enable_adaptive_lr,
        lr_strategy=args.lr_strategy,
        initial_lr=args.initial_lr,
        checkpoint_freq=args.checkpoint_freq,
        local_dir=args.local_dir,
    )
    
    trainer.train(num_workers=args.workers, use_phi=args.use_phi)


if __name__ == "__main__":
    main()