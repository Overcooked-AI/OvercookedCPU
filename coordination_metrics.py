# coordination_metrics.py
"""
Specialized coordination metrics for Overcooked MAPPO training
Tracks how well agents coordinate, specialize, and avoid conflicts
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from overcooked_ai_py.mdp.actions import Action


class CoordinationMetricsTracker:
    """
    Tracks advanced coordination metrics during episodes.
    Call update() at each step, get_metrics() at episode end.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for new episode."""
        # Action tracking
        self.agent_actions = {0: [], 1: []}
        self.joint_actions = []
        
        # Task tracking
        self.agent_task_times = {0: defaultdict(int), 1: defaultdict(int)}
        self.current_tasks = {0: "idle", 1: "idle"}
        
        # Coordination events
        self.collisions = 0
        self.pot_conflicts = 0  # Both agents at same pot
        self.handoffs = 0  # One places, other picks up
        
        # Efficiency tracking
        self.steps = 0
        self.soups_delivered = 0
        self.ingredients_picked = {0: 0, 1: 0}
        self.items_placed = {0: 0, 1: 0}
        self.cooking_started = 0
        
        # Spatial tracking
        self.agent_positions = {0: [], 1: []}
        self.position_overlaps = 0
        
        # State tracking
        self.prev_state = None
        self.prev_actions = None
    
    def update(self, state, actions: Dict[str, int], rewards: Dict[str, float], info: Dict):
        """
        Update metrics based on current step.
        
        Args:
            state: OvercookedState object
            actions: {"agent_0": action_idx, "agent_1": action_idx}
            rewards: Reward dict
            info: Info dict
        """
        self.steps += 1
        
        # Track actions
        action_0 = actions.get("agent_0", 4)
        action_1 = actions.get("agent_1", 4)
        self.agent_actions[0].append(action_0)
        self.agent_actions[1].append(action_1)
        self.joint_actions.append((action_0, action_1))
        
        # Track positions
        if hasattr(state, 'players'):
            pos_0 = state.players[0].position
            pos_1 = state.players[1].position
            self.agent_positions[0].append(pos_0)
            self.agent_positions[1].append(pos_1)
            
            # Check position overlap (collision)
            if pos_0 == pos_1:
                self.position_overlaps += 1
        
        # Track tasks from state
        if hasattr(state, 'players'):
            for i, player in enumerate(state.players):
                # Determine current task based on held object
                if player.held_object is None:
                    task = "idle"
                elif player.held_object.name == "onion":
                    task = "carrying_onion"
                elif player.held_object.name == "tomato":
                    task = "carrying_tomato"
                elif player.held_object.name == "dish":
                    task = "carrying_dish"
                elif player.held_object.name == "soup":
                    task = "carrying_soup"
                else:
                    task = "carrying_other"
                
                self.current_tasks[i] = task
                self.agent_task_times[i][task] += 1
        
        # Detect events
        if self.prev_state is not None:
            self._detect_coordination_events(self.prev_state, state, actions)
        
        # Track deliveries
        if rewards.get("agent_0", 0) > 10:  # Sparse reward for delivery
            self.soups_delivered += 1
        
        self.prev_state = state
        self.prev_actions = actions
    
    def _detect_coordination_events(self, prev_state, curr_state, actions):
        """Detect specific coordination events between steps."""
        
        # Detect item pickups
        for i in range(2):
            prev_held = prev_state.players[i].held_object
            curr_held = curr_state.players[i].held_object
            
            # Picked up ingredient
            if prev_held is None and curr_held is not None:
                if curr_held.name in ["onion", "tomato"]:
                    self.ingredients_picked[i] += 1
            
            # Placed item
            if prev_held is not None and curr_held is None:
                self.items_placed[i] += 1
        
        # Detect pot conflicts (both agents near same pot)
        if hasattr(curr_state, 'objects'):
            pots = [pos for pos, obj in curr_state.objects.items() 
                   if hasattr(obj, 'is_cooking') or hasattr(obj, 'is_ready')]
            
            if pots:
                for pot_pos in pots:
                    dist_0 = self._manhattan_distance(
                        curr_state.players[0].position, pot_pos
                    )
                    dist_1 = self._manhattan_distance(
                        curr_state.players[1].position, pot_pos
                    )
                    
                    if dist_0 <= 1 and dist_1 <= 1:
                        self.pot_conflicts += 1
                        break
        
        # Detect handoffs (one places, other immediately picks up at same location)
        # This is a simplified heuristic
        if self.items_placed[0] > 0 or self.items_placed[1] > 0:
            dist = self._manhattan_distance(
                curr_state.players[0].position,
                curr_state.players[1].position
            )
            if dist <= 1:
                self.handoffs += 1
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute and return all coordination metrics.
        
        Returns:
            Dictionary of metric_name -> value
        """
        if self.steps == 0:
            return {}
        
        metrics = {}
        
        # === Action Distribution Metrics ===
        action_names = [a.name for a in Action.ALL_ACTIONS]
        
        for agent_id in [0, 1]:
            actions = self.agent_actions[agent_id]
            total = len(actions)
            
            for i, action_name in enumerate(action_names):
                count = actions.count(i)
                metrics[f"agent_{agent_id}_action_{action_name}_pct"] = (
                    100 * count / total if total > 0 else 0
                )
        
        # === Task Specialization Metrics ===
        # How much time each agent spends on each task
        for agent_id in [0, 1]:
            total_time = sum(self.agent_task_times[agent_id].values())
            for task, time in self.agent_task_times[agent_id].items():
                pct = 100 * time / total_time if total_time > 0 else 0
                metrics[f"agent_{agent_id}_task_{task}_pct"] = pct
        
        # Task balance (difference in work distribution)
        task_diff = abs(
            len(self.agent_actions[0]) - len(self.agent_actions[1])
        )
        metrics["task_balance"] = 100 * (1 - task_diff / self.steps)
        
        # === Coordination Quality Metrics ===
        metrics["collisions_per_100_steps"] = 100 * self.position_overlaps / self.steps
        metrics["pot_conflicts_per_100_steps"] = 100 * self.pot_conflicts / self.steps
        metrics["handoffs_per_soup"] = (
            self.handoffs / self.soups_delivered if self.soups_delivered > 0 else 0
        )
        
        # === Efficiency Metrics ===
        metrics["soups_delivered"] = self.soups_delivered
        metrics["steps_per_soup"] = (
            self.steps / self.soups_delivered if self.soups_delivered > 0 else self.steps
        )
        
        total_ingredients = sum(self.ingredients_picked.values())
        metrics["ingredients_picked_per_soup"] = (
            total_ingredients / self.soups_delivered if self.soups_delivered > 0 else 0
        )
        
        # Ingredient balance (how equally agents share gathering)
        if total_ingredients > 0:
            ing_balance = min(
                self.ingredients_picked[0], self.ingredients_picked[1]
            ) / max(self.ingredients_picked[0], self.ingredients_picked[1])
            metrics["ingredient_gathering_balance"] = 100 * ing_balance
        else:
            metrics["ingredient_gathering_balance"] = 0
        
        # === Spatial Coverage ===
        # Measure how much of the map agents explore
        unique_pos_0 = len(set(self.agent_positions[0]))
        unique_pos_1 = len(set(self.agent_positions[1]))
        metrics["agent_0_spatial_coverage"] = unique_pos_0
        metrics["agent_1_spatial_coverage"] = unique_pos_1
        
        # Spatial overlap (how often agents visit same locations)
        overlap = len(
            set(self.agent_positions[0]).intersection(set(self.agent_positions[1]))
        )
        total_unique = len(
            set(self.agent_positions[0]).union(set(self.agent_positions[1]))
        )
        metrics["spatial_overlap_pct"] = (
            100 * overlap / total_unique if total_unique > 0 else 0
        )
        
        # === Action Synchronization ===
        # How often agents take same action simultaneously
        same_actions = sum(1 for a0, a1 in self.joint_actions if a0 == a1)
        metrics["action_synchronization_pct"] = 100 * same_actions / len(self.joint_actions)
        
        # Complementary actions (one moves, other interacts)
        move_actions = {1, 2, 3, 4}  # N, S, E, W
        interact_action = 5
        
        complementary = sum(
            1 for a0, a1 in self.joint_actions
            if (a0 in move_actions and a1 == interact_action) or
               (a1 in move_actions and a0 == interact_action)
        )
        metrics["complementary_actions_pct"] = 100 * complementary / len(self.joint_actions)
        
        return metrics
    
    def print_summary(self):
        """Print a human-readable summary of coordination metrics."""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("COORDINATION METRICS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Efficiency:")
        print(f"  Soups Delivered: {metrics.get('soups_delivered', 0):.0f}")
        print(f"  Steps per Soup: {metrics.get('steps_per_soup', 0):.1f}")
        print(f"  Ingredients per Soup: {metrics.get('ingredients_picked_per_soup', 0):.1f}")
        
        print(f"\n🤝 Coordination Quality:")
        print(f"  Collisions: {metrics.get('collisions_per_100_steps', 0):.2f} per 100 steps")
        print(f"  Pot Conflicts: {metrics.get('pot_conflicts_per_100_steps', 0):.2f} per 100 steps")
        print(f"  Handoffs per Soup: {metrics.get('handoffs_per_soup', 0):.2f}")
        
        print(f"\n⚖️  Task Balance:")
        print(f"  Overall Balance: {metrics.get('task_balance', 0):.1f}%")
        print(f"  Ingredient Gathering Balance: {metrics.get('ingredient_gathering_balance', 0):.1f}%")
        
        print(f"\n🗺️  Spatial Coverage:")
        print(f"  Agent 0 Coverage: {metrics.get('agent_0_spatial_coverage', 0):.0f} tiles")
        print(f"  Agent 1 Coverage: {metrics.get('agent_1_spatial_coverage', 0):.0f} tiles")
        print(f"  Spatial Overlap: {metrics.get('spatial_overlap_pct', 0):.1f}%")
        
        print(f"\n🎭 Action Patterns:")
        print(f"  Action Synchronization: {metrics.get('action_synchronization_pct', 0):.1f}%")
        print(f"  Complementary Actions: {metrics.get('complementary_actions_pct', 0):.1f}%")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    print("Coordination Metrics Tracker - Ready to use!")
    print("\nUsage:")
    print("  tracker = CoordinationMetricsTracker()")
    print("  tracker.update(state, actions, rewards, info)")
    print("  metrics = tracker.get_metrics()")
    print("  tracker.print_summary()")