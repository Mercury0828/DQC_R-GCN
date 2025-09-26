"""
Basic test script to verify Phase 1 components work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_environment():
    """Test environment creation and basic operations."""
    print("Testing Environment...")
    
    from data.quantum_problem_generator import QuantumProblemGenerator
    from environment import DQCEnvironment
    
    # Generate small problem
    generator = QuantumProblemGenerator(
        num_gates=10,
        num_qpus=2,
        num_qubits=6,
        qpu_capacity=3,
        time_horizon=20,
        seed=42
    )
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # Create environment
    env = DQCEnvironment(problem_data)
    
    # Test reset
    state = env.reset()
    print(f"  ✓ Environment reset successful, state shape: {state.shape}")
    
    # Test getting valid actions
    valid_actions = env.get_valid_actions()
    print(f"  ✓ Valid actions: {len(valid_actions['map'])} map, {len(valid_actions['schedule'])} schedule")
    
    # Test step
    if valid_actions['map']:
        action = ('map', valid_actions['map'][0])
        next_state, reward, done, info = env.step(action)
        print(f"  ✓ Step successful, reward: {reward:.3f}")
    
    return True

def test_models():
    """Test model creation and forward pass."""
    print("\nTesting Models...")
    
    from models import RGCN, PolicyNetwork, ValueNetwork
    from environment.state import DQCState
    from data.quantum_problem_generator import QuantumProblemGenerator
    
    # Generate problem
    generator = QuantumProblemGenerator(
        num_gates=10,
        num_qpus=2,
        num_qubits=6,
        qpu_capacity=3,
        time_horizon=20,
        seed=42
    )
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # Create state
    state = DQCState(problem_data)
    state_repr = state.get_state_representation()
    
    # Create models
    config = {
        'hidden_dim': 32,
        'num_layers': 2,
        'num_bases': 4,
        'dropout': 0.0
    }
    
    rgcn = RGCN(config)
    policy_net = PolicyNetwork(rgcn, config)
    value_net = ValueNetwork(rgcn, config)
    
    print("  ✓ Models created successfully")
    
    # Test forward pass
    try:
        # Test value network
        value = value_net(state_repr)
        print(f"  ✓ Value network forward pass, output: {value.item():.3f}")
        
        # Test policy network
        valid_actions = {'map': [(0, 0), (1, 1)], 'schedule': []}
        probs, logits, info = policy_net.forward(state_repr, valid_actions)
        print(f"  ✓ Policy network forward pass, action probs shape: {probs.shape}")
    except Exception as e:
        print(f"  ✗ Model forward pass failed: {e}")
        return False
    
    return True

def test_ppo_basic():
    """Test basic PPO functionality."""
    print("\nTesting PPO...")
    
    from data.quantum_problem_generator import QuantumProblemGenerator
    from environment import DQCEnvironment
    from models import RGCN, PolicyNetwork, ValueNetwork
    from algorithms.ppo import PPO
    
    # Setup
    generator = QuantumProblemGenerator(
        num_gates=10,
        num_qpus=2,
        num_qubits=6,
        qpu_capacity=3,
        time_horizon=20,
        seed=42
    )
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    env = DQCEnvironment(problem_data)
    
    config = {'hidden_dim': 32, 'num_layers': 2}
    rgcn = RGCN(config)
    policy_net = PolicyNetwork(rgcn, config)
    value_net = ValueNetwork(rgcn, config)
    
    ppo_config = {
        'learning_rate': 1e-3,
        'rollout_length': 20,
        'minibatch_size': 5,
        'num_epochs': 1
    }
    
    ppo = PPO(policy_net, value_net, ppo_config)
    print("  ✓ PPO created successfully")
    
    # Test rollout collection
    try:
        ppo.collect_rollouts(env, 10)
        print(f"  ✓ Collected {len(ppo.rollout_buffer)} transitions")
    except Exception as e:
        print(f"  ✗ Rollout collection failed: {e}")
        return False
    
    return True

def main():
    """Run all basic tests."""
    print("="*50)
    print("PHASE 1 BASIC TESTS")
    print("="*50)
    
    results = []
    
    # Test each component
    results.append(("Environment", test_environment()))
    results.append(("Models", test_models()))
    results.append(("PPO", test_ppo_basic()))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All basic tests passed! Ready for full training.")
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)