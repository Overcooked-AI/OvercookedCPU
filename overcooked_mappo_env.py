# overcooked_mappo_env.py
"""
RLlib MultiAgentEnv wrapper for Overcooked-AI
Uses PADDED lossless_state_encoding for universal compatibility across map sizes.
Fixed for Reward Shaping (Motion Planner API compatibility).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
# Import MotionPlanner for reward shaping
from overcooked_ai_py.planning.planners import MotionPlanner

# --- CONFIGURATION ---
# Define a maximum map size that covers all layouts you intend to use.
MAX_WIDTH = 10 
MAX_HEIGHT = 10

class OvercookedMAPPOEnv(MultiAgentEnv):
    """
    Wrapper that pads observations to a fixed size (MAX_WIDTH x MAX_HEIGHT).
    Allows agents to be transferred between maps of different sizes.
    """
    
    def __init__(self, env_config):
        super().__init__()
        
        self.layout_name = env_config.get("layout_name", "cramped_room")
        self.horizon = env_config.get("horizon", 400)
        self.use_phi = env_config.get("use_phi", False)
        self.reward_shaping_factor = env_config.get("reward_shaping_factor", 1.0)
        
        self.base_mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        
        # --- FIX: Explicitly Load Motion Planner ---
        self.motion_planner = None
        if self.use_phi:
            try:
                # Removed 'event_based_goals' argument which caused the error
                self.motion_planner = MotionPlanner.from_pickle_or_compute(
                    self.base_mdp,
                    counter_goals=[]
                )
            except Exception as e:
                print(f"Warning: Could not load MotionPlanner: {e}")
                self.use_phi = False # Disable shaping if MP fails
        
        self.env = OvercookedEnv.from_mdp(self.base_mdp, horizon=self.horizon, info_level=0)
        
        self.agents = ["agent_0", "agent_1"]
        self._agent_ids = set(self.agents)
        
        # --- CALCULATE PADDED SHAPE ---
        dummy_state = self.env.state
        # Get the actual shape (Channels, H, W)
        obs_tuple = self.base_mdp.lossless_state_encoding(dummy_state, horizon=self.horizon)
        real_shape = obs_tuple[0].shape
        self.num_channels = real_shape[0]
        
        # The Observation Space is now FIXED based on MAX dimensions
        self.padded_shape = (self.num_channels * MAX_HEIGHT * MAX_WIDTH,)
        
        self._single_observation_space = spaces.Box(
            low=0, high=1, shape=self.padded_shape, dtype=np.float32
        )
        self._single_action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        
        self.observation_space = spaces.Dict({
            agent_id: self._single_observation_space for agent_id in self.agents
        })
        self.action_space = spaces.Dict({
            agent_id: self._single_action_space for agent_id in self.agents
        })
        
        self.action_to_overcooked = {i: Action.ALL_ACTIONS[i] for i in range(len(Action.ALL_ACTIONS))}
        
        self.timestep = 0
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        self.timestep = 0
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        return self._get_obs(), {}
    
    def step(self, action_dict):
        joint_action = (
            self.action_to_overcooked[action_dict.get("agent_0", 4)],
            self.action_to_overcooked[action_dict.get("agent_1", 4)]
        )
        
        next_state, sparse_reward, done, info = self.env.step(joint_action)
        
        shaped_reward = 0
        if self.use_phi and self.motion_planner:
            shaped_reward = self.reward_shaping_factor * self._get_potential()
        
        total_reward = sparse_reward + shaped_reward
        self.cumulative_sparse_rewards += sparse_reward
        self.cumulative_shaped_rewards += shaped_reward
        self.timestep += 1
        
        obs_dict = self._get_obs()
        
        reward_dict = {"agent_0": total_reward, "agent_1": total_reward}
        terminateds = {"agent_0": done, "agent_1": done, "__all__": done}
        truncateds = {"agent_0": False, "agent_1": False, "__all__": False}
        info_dict = {"agent_0": info, "agent_1": info}
        
        if done:
            stats = {
                "sparse_reward": self.cumulative_sparse_rewards,
                "shaped_reward": self.cumulative_shaped_rewards,
                "total_reward": self.cumulative_sparse_rewards + self.cumulative_shaped_rewards,
            }
            info_dict["agent_0"]["episode"] = stats
            info_dict["agent_1"]["episode"] = stats
        
        return obs_dict, reward_dict, terminateds, truncateds, info_dict
    
    def _get_obs(self):
        obs_tuple = self.base_mdp.lossless_state_encoding(self.env.state, horizon=self.horizon)
        obs_dict = {}
        
        for i, agent_id in enumerate(self.agents):
            real_obs = obs_tuple[i]
            padded_obs = np.zeros((self.num_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
            current_channels, current_h, current_w = real_obs.shape
            safe_h = min(current_h, MAX_HEIGHT)
            safe_w = min(current_w, MAX_WIDTH)
            padded_obs[:current_channels, :safe_h, :safe_w] = real_obs[:current_channels, :safe_h, :safe_w]
            obs_dict[agent_id] = padded_obs.flatten()
            
        return obs_dict

    def _get_potential(self):
        # Use the self.motion_planner we loaded in __init__
        return self.base_mdp.potential_function(
            self.env.state,
            mp=self.motion_planner,
            gamma=0.99
        )
        
    def render(self, mode='human'): pass
    def close(self): pass

def make_overcooked_env(env_config):
    return OvercookedMAPPOEnv(env_config)

if __name__ == "__main__":
    print("Testing Overcooked MAPPO Environment...")
    test_config = {
        "layout_name": "cramped_room",
        "horizon": 400,
        "use_phi": True, # Test WITH reward shaping
    }
    env = OvercookedMAPPOEnv(test_config)
    obs, info = env.reset()
    if env.use_phi:
        print("Success: Motion Planner loaded and Reward Shaping is ACTIVE!")
    else:
        print("Failure: Reward Shaping is DISABLED due to errors.")
    print(f"Observation shape: {obs['agent_0'].shape}")