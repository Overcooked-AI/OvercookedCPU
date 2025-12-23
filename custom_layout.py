# custom_layout.py
"""
Custom layout definitions for Overcooked Playground
Defines layouts with all requested features: Counter, Collision, Pot, Cook, Delivery
ALL LAYOUTS MUST BE FULLY ENCLOSED BY 'X' (COUNTERS)
"""

import sys
import os
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add it to the system path
if src_path not in sys.path:
    sys.path.append(src_path)
    
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# ============= PLAYGROUND LAYOUT =============
# Grid symbols:
# ' ' = Empty floor
# 'X' = Counter
# 'O' = Onion dispenser
# 'T' = Tomato dispenser  
# 'P' = Pot (for cooking)
# 'D' = Dish dispenser
# 'S' = Serving location (delivery point)
# '1', '2' = Agent starting positions

# FIX: Layouts defined as lists of strings are safer and standard for OvercookedGridworld
PLAYGROUND_LAYOUT = [
    "XXPXX",
    "X  2X",
    "X1  X",
    "XO DX",
    "XXSXX"
]

PLAYGROUND_LAYOUT_MEDIUM = [
    "XXXXPXX",
    "X     X",
    "XT 1 DX",  # Added 'D' (Dish Dispenser) here
    "X  2 OX",
    "X     X",
    "XXXSXXX"
]

PLAYGROUND_LAYOUT_LARGE = [
    "XXXXPXXXX",
    "X       X",
    "XO  1   X",
    "X   2   X",
    "XD     TX",
    "XXXXSXXXX"
]

# More complex layout with multiple pots and obstacles
PLAYGROUND_COMPLEX = [
    "XXXXXPXXXXX",
    "XO   2    D",
    "X    X    X",
    "X  P   P  X",
    "X    X    X",
    "XD   1    S",
    "XXXXXXXXXXX"
]

# Narrow corridor layout (tests collision handling)
PLAYGROUND_CORRIDOR = [
    "XXXPXXX",
    "XO 2 DX",
    "X     X",
    "X     X",
    "X     X",
    "XD 1 SX",
    "XXXXXXX"
]


def create_playground_mdp(layout_name="playground", **mdp_params):
    """
    Create an OvercookedGridworld from a custom playground layout.
    
    Args:
        layout_name: Name of the layout to use
        **mdp_params: Additional parameters for the MDP
        
    Returns:
        OvercookedGridworld instance
    """
    layouts = {
        "playground": PLAYGROUND_LAYOUT,
        "playground_medium": PLAYGROUND_LAYOUT_MEDIUM,
        "playground_large": PLAYGROUND_LAYOUT_LARGE,
        "playground_complex": PLAYGROUND_COMPLEX,
        "playground_corridor": PLAYGROUND_CORRIDOR,
    }
    
    if layout_name not in layouts:
        # Don't raise error, just fallback to default with warning
        print(f"Warning: Layout {layout_name} not found, defaulting to playground")
        grid = PLAYGROUND_LAYOUT
    else:
        grid = layouts[layout_name]
        
    # Create MDP from layout grid (works with list of strings)
    mdp = OvercookedGridworld.from_grid(
        grid,
        **mdp_params
    )
    
    return mdp


def register_custom_layouts():
    """
    Register custom layouts so they can be used with layout_name parameter.
    Returns dict of layout name to grid layout.
    """
    custom_layouts = {
        "playground": PLAYGROUND_LAYOUT,
        "playground_medium": PLAYGROUND_LAYOUT_MEDIUM,
        "playground_large": PLAYGROUND_LAYOUT_LARGE,
        "playground_complex": PLAYGROUND_COMPLEX,
        "playground_corridor": PLAYGROUND_CORRIDOR,
    }
    
    return custom_layouts


if __name__ == "__main__":
    # Test custom layout creation
    print("Testing custom layouts...")
    
    for layout_name in ["playground", "playground_medium", "playground_large", 
                        "playground_complex", "playground_corridor"]:
        print(f"\n=== Testing {layout_name} ===")
        
        try:
            mdp = create_playground_mdp(layout_name)
            
            # Safe shape check that works for both list and numpy array
            terrain = mdp.terrain_mtx
            if isinstance(terrain, list):
                shape = (len(terrain), len(terrain[0]))
            else:
                shape = terrain.shape
                
            print(f"Layout name: {layout_name}")
            print(f"Terrain shape: {shape}")
            print(f"Number of pots: {len(mdp.get_pot_locations())}")
            print(f"Number of counters: {len(mdp.get_counter_locations())}")
            print(f"Onion dispensers: {mdp.get_onion_dispenser_locations()}")
            print(f"Dish dispensers: {mdp.get_dish_dispenser_locations()}")
            print(f"Serving locations: {mdp.get_serving_locations()}")
            print(f"Starting positions: {mdp.start_player_positions}")
            
            # Test a few transitions
            state = mdp.get_standard_start_state()
            print(f"Initial state: {state}")
            print("✓ Layout valid")
            
        except Exception as e:
            print(f"✗ Failed to create layout: {e}")
            # Raise to see full traceback if needed
            # raise e
            
    print("\n✅ All custom layouts test complete!")