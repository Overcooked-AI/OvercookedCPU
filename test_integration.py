#!/usr/bin/env python3
"""
test_integration.py
Quick integration test to verify all new scripts work with existing code
"""

import sys
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test existing files
        print("  - Testing existing files...")
        from config import ENV_CONFIG, TRAINING_CONFIG
        from overcooked_mappo_env import OvercookedMAPPOEnv, make_overcooked_env
        from custom_layout import create_playground_mdp
        print("    ✓ Existing files import successfully")
        
        # Test new files
        print("  - Testing new files...")
        from coordination_metrics import CoordinationMetricsTracker
        from training_monitor import TrainingMonitor
        print("    ✓ New analysis files import successfully")
        
        return True
        
    except ImportError as e:
        print(f"    ✗ Import failed: {e}")
        return False


def test_environment():
    """Test that environment can be created."""
    print("\n🧪 Testing environment creation...")
    
    try:
        from overcooked_mappo_env import OvercookedMAPPOEnv
        from config import ENV_CONFIG
        
        # Test with built-in layout
        env_config = ENV_CONFIG.copy()
        env_config["layout_name"] = "cramped_room"
        env = OvercookedMAPPOEnv(env_config)
        
        obs, info = env.reset()
        print(f"    ✓ Environment created: {env_config['layout_name']}")
        print(f"    ✓ Observation shape: {obs['agent_0'].shape}")
        print(f"    ✓ Action space: {env.action_space}")
        
        # Test a few steps
        actions = {"agent_0": 0, "agent_1": 0}
        for i in range(3):
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, rewards, terminated, truncated, infos = step_result
            else:
                obs, rewards, dones, infos = step_result
        
        print("    ✓ Environment stepping works")
        return True
        
    except Exception as e:
        print(f"    ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_layouts():
    """Test custom layout creation."""
    print("\n🧪 Testing custom layouts...")
    
    try:
        from custom_layout import create_playground_mdp
        
        layouts_to_test = ["playground", "playground_medium", "playground_large"]
        
        for layout_name in layouts_to_test:
            mdp = create_playground_mdp(layout_name)
            print(f"    ✓ {layout_name}: ({len(mdp.terrain_mtx)}, {len(mdp.terrain_mtx[0])})")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Custom layout test failed: {e}")
        return False


def test_coordination_metrics():
    """Test coordination metrics tracker."""
    print("\n🧪 Testing coordination metrics...")
    
    try:
        from coordination_metrics import CoordinationMetricsTracker
        
        tracker = CoordinationMetricsTracker()
        tracker.reset()
        
        # Simulate some updates
        actions = {"agent_0": 1, "agent_1": 2}
        rewards = {"agent_0": 0.0, "agent_1": 0.0}
        info = {}
        
        for _ in range(10):
            tracker.update(None, actions, rewards, info)
        
        metrics = tracker.get_metrics()
        print(f"    ✓ Coordination metrics computed: {len(metrics)} metrics")
        print(f"    ✓ Sample metrics: {list(metrics.keys())[:3]}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Coordination metrics test failed: {e}")
        return False


def test_training_monitor():
    """Test training monitor callback."""
    print("\n🧪 Testing training monitor...")
    
    try:
        from training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor()
        print("    ✓ Training monitor instantiated")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Training monitor test failed: {e}")
        return False


def test_adaptive_trainer():
    """Test adaptive trainer components."""
    print("\n🧪 Testing adaptive trainer...")
    
    try:
        from adaptive_trainer import CurriculumSchedule, AdaptiveLearningRate
        
        # Test curriculum
        curriculum = CurriculumSchedule("progressive")
        layout = curriculum.get_layout_for_iteration(0, 100)
        print(f"    ✓ Curriculum schedule: {layout}")
        
        # Test adaptive LR
        lr_scheduler = AdaptiveLearningRate(initial_lr=5e-4)
        lr = lr_scheduler.get_lr(50, 100)
        print(f"    ✓ Adaptive LR computed: {lr:.6f}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Adaptive trainer test failed: {e}")
        return False


def test_ray_integration():
    """Test Ray initialization (quick test)."""
    print("\n🧪 Testing Ray integration...")
    
    try:
        import ray
        
        if ray.is_initialized():
            print("    ℹ Ray already initialized")
        else:
            ray.init(ignore_reinit_error=True, num_cpus=2)
            print("    ✓ Ray initialized successfully")
        
        # Test environment registration
        from ray.tune.registry import register_env
        from overcooked_mappo_env import make_overcooked_env
        
        register_env("test_overcooked", make_overcooked_env)
        print("    ✓ Environment registered with Ray")
        
        ray.shutdown()
        print("    ✓ Ray shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Ray integration test failed: {e}")
        return False


def test_file_structure():
    """Check that all required files exist."""
    print("\n🧪 Checking file structure...")
    
    required_files = {
        "existing": [
            "train_mappo.py",
            "overcooked_mappo_env.py",
            "custom_layout.py",
            "config.py",
        ],
        "new": [
            "coordination_metrics.py",
            "training_monitor.py",
            "adaptive_trainer.py",
            "analyze_behavior.py",
            "tune_hyperparams.py",
        ]
    }
    
    all_exist = True
    
    for category, files in required_files.items():
        print(f"  - Checking {category} files...")
        for filename in files:
            if os.path.exists(filename):
                print(f"    ✓ {filename}")
            else:
                print(f"    ✗ {filename} NOT FOUND")
                all_exist = False
    
    return all_exist


def main():
    """Run all integration tests."""
    print("="*70)
    print("  MAPPO Overcooked - Integration Test Suite")
    print("="*70)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Custom Layouts", test_custom_layouts),
        ("Coordination Metrics", test_coordination_metrics),
        ("Training Monitor", test_training_monitor),
        ("Adaptive Trainer", test_adaptive_trainer),
        ("Ray Integration", test_ray_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} - {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your integration is working correctly.")
        print("\nNext steps:")
        print("  1. Run quick training demo:")
        print("     python train_mappo.py --iterations 10 --workers 2")
        print("\n  2. Run adaptive training demo:")
        print("     python adaptive_trainer.py --iterations 10 --workers 2")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())