# analyze_behavior.py
"""
Behavioral analysis tools for trained MAPPO agents
Generates visualizations, heatmaps, and detailed behavior reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from overcooked_ai_py.mdp.actions import Action
from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
from config import ENV_CONFIG
from coordination_metrics import CoordinationMetricsTracker


class BehaviorAnalyzer:
    """
    Analyze trained agent behavior across multiple episodes.
    Generates visualizations and detailed reports.
    """
    
    def __init__(self, checkpoint_path: str, layout_name: str = "cramped_room"):
        """
        Args:
            checkpoint_path: Path to trained checkpoint
            layout_name: Layout to analyze
        """
        self.checkpoint_path = checkpoint_path
        self.layout_name = layout_name
        
        # Initialize Ray and load agent
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)
        
        register_env("overcooked_mappo", make_overcooked_env)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        self.agent = Algorithm.from_checkpoint(checkpoint_path)
        
        # Create environment
        env_config = ENV_CONFIG.copy()
        env_config["layout_name"] = layout_name
        self.env = OvercookedMAPPOEnv(env_config)
        
        print(f"Analyzing layout: {layout_name}")
        print(f"Map size: {self.env.base_mdp.terrain_mtx.shape}")
    
    def collect_episodes(self, num_episodes: int = 50) -> List[Dict]:
        """
        Collect episode trajectories for analysis.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of episode data dictionaries
        """
        print(f"\nCollecting {num_episodes} episodes...")
        episodes = []
        
        for ep in range(num_episodes):
            episode_data = {
                "states": [],
                "actions": {0: [], 1: []},
                "positions": {0: [], 1: []},
                "held_objects": {0: [], 1: []},
                "rewards": [],
                "coordination_metrics": None,
            }
            
            obs, _ = self.env.reset()
            done = False
            step = 0
            tracker = CoordinationMetricsTracker()
            
            while not done:
                # Store state
                episode_data["states"].append(self.env.env.state)
                
                # Get actions
                actions = {}
                for agent_id in ["agent_0", "agent_1"]:
                    action = self.agent.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy"
                    )
                    actions[agent_id] = action
                    episode_data["actions"][int(agent_id[-1])].append(action)
                
                # Store positions and held objects
                state = self.env.env.state
                if hasattr(state, 'players'):
                    for i, player in enumerate(state.players):
                        episode_data["positions"][i].append(player.position)
                        episode_data["held_objects"][i].append(
                            player.held_object.name if player.held_object else None
                        )
                
                # Step environment
                step_result = self.env.step(actions)
                
                if len(step_result) == 5:
                    obs, rewards, terminated, truncated, infos = step_result
                    done = terminated["__all__"] or truncated["__all__"]
                else:
                    obs, rewards, dones, infos = step_result
                    done = dones["__all__"] if isinstance(dones, dict) else dones
                
                episode_data["rewards"].append(rewards["agent_0"])
                
                # Update coordination tracker
                tracker.update(state, actions, rewards, infos)
                
                step += 1
            
            # Store coordination metrics
            episode_data["coordination_metrics"] = tracker.get_metrics()
            episode_data["total_reward"] = sum(episode_data["rewards"])
            episode_data["length"] = step
            
            episodes.append(episode_data)
            
            if (ep + 1) % 10 == 0:
                print(f"  Collected {ep + 1}/{num_episodes} episodes")
        
        print("✓ Episode collection complete\n")
        return episodes
    
    def generate_heatmaps(self, episodes: List[Dict], save_dir: str = "./analysis"):
        """
        Generate position heatmaps for both agents.
        
        Args:
            episodes: List of episode data
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get map dimensions
        terrain = self.env.base_mdp.terrain_mtx
        height, width = terrain.shape
        
        # Create heatmaps
        heatmap_0 = np.zeros((height, width))
        heatmap_1 = np.zeros((height, width))
        
        for episode in episodes:
            for pos in episode["positions"][0]:
                heatmap_0[pos[1], pos[0]] += 1
            for pos in episode["positions"][1]:
                heatmap_1[pos[1], pos[0]] += 1
        
        # Normalize
        heatmap_0 = heatmap_0 / (heatmap_0.sum() + 1e-8) * 100
        heatmap_1 = heatmap_1 / (heatmap_1.sum() + 1e-8) * 100
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Agent 0 heatmap
        sns.heatmap(heatmap_0, annot=True, fmt=".1f", cmap="YlOrRd",
                   ax=axes[0], cbar_kws={'label': 'Visit Frequency (%)'})
        axes[0].set_title("Agent 0 Position Heatmap", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("X Position")
        axes[0].set_ylabel("Y Position")
        
        # Agent 1 heatmap
        sns.heatmap(heatmap_1, annot=True, fmt=".1f", cmap="YlGnBu",
                   ax=axes[1], cbar_kws={'label': 'Visit Frequency (%)'})
        axes[1].set_title("Agent 1 Position Heatmap", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("X Position")
        axes[1].set_ylabel("Y Position")
        
        # Combined heatmap
        combined = heatmap_0 + heatmap_1
        sns.heatmap(combined, annot=True, fmt=".1f", cmap="viridis",
                   ax=axes[2], cbar_kws={'label': 'Combined Frequency (%)'})
        axes[2].set_title("Combined Position Heatmap", fontsize=14, fontweight='bold')
        axes[2].set_xlabel("X Position")
        axes[2].set_ylabel("Y Position")
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"heatmap_{self.layout_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved: {save_path}")
        plt.close()
    
    def analyze_action_distributions(self, episodes: List[Dict], save_dir: str = "./analysis"):
        """
        Analyze and visualize action distributions.
        
        Args:
            episodes: List of episode data
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        action_names = [a.name for a in Action.ALL_ACTIONS]
        
        # Count actions for each agent
        action_counts = {0: Counter(), 1: Counter()}
        
        for episode in episodes:
            for agent_id in [0, 1]:
                for action in episode["actions"][agent_id]:
                    action_counts[agent_id][action] += 1
        
        # Convert to percentages
        action_pcts = {0: {}, 1: {}}
        for agent_id in [0, 1]:
            total = sum(action_counts[agent_id].values())
            for action_idx in range(len(action_names)):
                count = action_counts[agent_id].get(action_idx, 0)
                action_pcts[agent_id][action_names[action_idx]] = (
                    100 * count / total if total > 0 else 0
                )
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, agent_id in enumerate([0, 1]):
            actions = list(action_pcts[agent_id].keys())
            values = list(action_pcts[agent_id].values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
            axes[i].bar(actions, values, color=colors, edgecolor='black')
            axes[i].set_title(f"Agent {agent_id} Action Distribution",
                            fontsize=14, fontweight='bold')
            axes[i].set_xlabel("Action")
            axes[i].set_ylabel("Frequency (%)")
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"actions_{self.layout_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Action distribution saved: {save_path}")
        plt.close()
        
        # Print summary
        print("\nAction Distribution Summary:")
        for agent_id in [0, 1]:
            print(f"\n  Agent {agent_id}:")
            for action_name, pct in sorted(
                action_pcts[agent_id].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    {action_name:12s}: {pct:5.1f}%")
    
    def analyze_task_specialization(self, episodes: List[Dict], save_dir: str = "./analysis"):
        """
        Analyze how agents specialize in different tasks.
        
        Args:
            episodes: List of episode data
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Track what each agent carries
        carried_items = {0: Counter(), 1: Counter()}
        
        for episode in episodes:
            for agent_id in [0, 1]:
                for item in episode["held_objects"][agent_id]:
                    if item is not None:
                        carried_items[agent_id][item] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        item_types = set()
        for agent_id in [0, 1]:
            item_types.update(carried_items[agent_id].keys())
        item_types = sorted(list(item_types))
        
        x = np.arange(len(item_types))
        width = 0.35
        
        counts_0 = [carried_items[0].get(item, 0) for item in item_types]
        counts_1 = [carried_items[1].get(item, 0) for item in item_types]
        
        # Normalize to percentages
        total_0 = sum(counts_0) or 1
        total_1 = sum(counts_1) or 1
        pcts_0 = [100 * c / total_0 for c in counts_0]
        pcts_1 = [100 * c / total_1 for c in counts_1]
        
        ax.bar(x - width/2, pcts_0, width, label='Agent 0', color='#FF6B6B', edgecolor='black')
        ax.bar(x + width/2, pcts_1, width, label='Agent 1', color='#4ECDC4', edgecolor='black')
        
        ax.set_xlabel('Item Type', fontsize=12)
        ax.set_ylabel('Time Carrying (%)', fontsize=12)
        ax.set_title('Task Specialization: Item Carrying', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(item_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"specialization_{self.layout_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Task specialization saved: {save_path}")
        plt.close()
    
    def generate_report(self, episodes: List[Dict], save_dir: str = "./analysis"):
        """
        Generate comprehensive behavior analysis report.
        
        Args:
            episodes: List of episode data
            save_dir: Directory to save report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Aggregate metrics
        metrics = defaultdict(list)
        for episode in episodes:
            if episode["coordination_metrics"]:
                for key, value in episode["coordination_metrics"].items():
                    metrics[key].append(value)
        
        # Compute statistics
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("BEHAVIORAL ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append(f"\nCheckpoint: {self.checkpoint_path}")
        report_lines.append(f"Layout: {self.layout_name}")
        report_lines.append(f"Episodes Analyzed: {len(episodes)}")
        report_lines.append("\n" + "="*70)
        
        # Episode statistics
        rewards = [ep["total_reward"] for ep in episodes]
        lengths = [ep["length"] for ep in episodes]
        
        report_lines.append("\n📊 EPISODE STATISTICS")
        report_lines.append("-"*70)
        report_lines.append(f"  Mean Reward:      {np.mean(rewards):8.2f} ± {np.std(rewards):6.2f}")
        report_lines.append(f"  Median Reward:    {np.median(rewards):8.2f}")
        report_lines.append(f"  Min/Max Reward:   {np.min(rewards):8.2f} / {np.max(rewards):8.2f}")
        report_lines.append(f"  Mean Length:      {np.mean(lengths):8.1f} ± {np.std(lengths):6.1f}")
        
        # Coordination metrics
        report_lines.append("\n🤝 COORDINATION METRICS")
        report_lines.append("-"*70)
        
        key_metrics = [
            "soups_delivered",
            "steps_per_soup",
            "collisions_per_100_steps",
            "task_balance",
            "ingredient_gathering_balance",
        ]
        
        for key in key_metrics:
            if key in metrics and len(metrics[key]) > 0:
                values = metrics[key]
                report_lines.append(
                    f"  {key:30s}: {np.mean(values):8.2f} ± {np.std(values):6.2f}"
                )
        
        # Save report
        report_path = os.path.join(save_dir, f"report_{self.layout_name}.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        print('\n'.join(report_lines))
        print(f"\n✓ Report saved: {report_path}")
    
    def analyze_all(self, num_episodes: int = 50, save_dir: str = "./analysis"):
        """
        Run complete behavioral analysis pipeline.
        
        Args:
            num_episodes: Number of episodes to analyze
            save_dir: Directory to save all outputs
        """
        print("\n" + "="*70)
        print("STARTING BEHAVIORAL ANALYSIS")
        print("="*70)
        
        # Collect episodes
        episodes = self.collect_episodes(num_episodes)
        
        # Generate all visualizations and reports
        print("\nGenerating visualizations...")
        self.generate_heatmaps(episodes, save_dir)
        self.analyze_action_distributions(episodes, save_dir)
        self.analyze_task_specialization(episodes, save_dir)
        
        print("\nGenerating report...")
        self.generate_report(episodes, save_dir)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print(f"All outputs saved to: {save_dir}")
        print("="*70)
    
    def cleanup(self):
        """Cleanup Ray resources."""
        ray.shutdown()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trained MAPPO agent behavior")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    parser.add_argument("--layout", type=str, default="cramped_room",
                       help="Layout to analyze")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes to analyze")
    parser.add_argument("--save-dir", type=str, default="./analysis",
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    analyzer = BehaviorAnalyzer(args.checkpoint, args.layout)
    analyzer.analyze_all(args.episodes, args.save_dir)
    analyzer.cleanup()


if __name__ == "__main__":
    main()