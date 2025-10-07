# main_improved.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import yaml
from pathlib import Path

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from training.two_stage_trainer import TwoStageTrainer

def main():
    """改进的主训练脚本"""
    print("="*60)
    print("IMPROVED DQC TRAINING - PLACEMENT FOCUSED")
    print("="*60)
    
    # 设置种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载配置
    with open('configs/training_config_v2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建问题
    problem_config = config['problem']['small']
    generator = QuantumProblemGenerator(**problem_config, seed=42)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # 创建环境（使用新的奖励配置）
    env_config = {
        'max_steps': 200,
        **config['imitation_learning']['reward_weights']
    }
    env = DQCEnvironment(problem_data, env_config)
    
    # 创建网络
    model_config = config['model']
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    print(f"\nModel architecture:")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Num heads: {model_config['num_heads']}")
    
    # 创建PPO
    ppo = PPO(policy_net, value_net, config['ppo'])
    
    # 创建两阶段训练器
    trainer_config = {
        'bc_lr': config['imitation_learning']['bc_pretrain_lr'],
        'reward_config': config['imitation_learning']['reward_weights']
    }
    
    trainer = TwoStageTrainer(env, policy_net, value_net, ppo, trainer_config)
    
    # 阶段1：Placement训练
    if config['training']['stage1']['enabled']:
        print("\nStarting Stage 1: Placement Training")
        trainer.stage1_placement_training(
            num_episodes=config['training']['stage1']['episodes']
        )
        
        # 保存阶段1模型
        torch.save({
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'config': config
        }, 'stage1_placement_model.pt')
        print("Stage 1 model saved!")
    
    # 阶段2：完整训练
    if config['training']['stage2']['enabled']:
        print("\nStarting Stage 2: Full Policy Training")
        trainer.stage2_full_training(
            timesteps=config['training']['stage2']['timesteps']
        )
        
        # 保存最终模型
        torch.save({
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'config': config,
            'final_stats': dict(trainer.stats)
        }, 'improved_model.pt')
        print("Final model saved!")
    
    # 最终评估
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # 测试placement质量
    env.reset()
    comm_costs = []
    
    for _ in range(5):
        env.reset()
        
        # 执行mapping
        while len(env.state.mapped_qubits) < env.state.num_qubits:
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions['map']:
                break
            
            with torch.no_grad():
                action, _ = policy_net.get_action(state_repr, valid_actions, deterministic=True)
            
            env.step(action)
        
        # 计算通信成本
        comm_cost = trainer._calculate_communication_cost()
        comm_costs.append(comm_cost)
    
    avg_comm_cost = np.mean(comm_costs)
    print(f"\nAverage communication cost: {avg_comm_cost:.2f}")
    print(f"Expert baseline: ~30.0")
    
    if avg_comm_cost < 35:
        print("✅ Excellent placement quality!")
    elif avg_comm_cost < 40:
        print("✓ Good placement quality")
    else:
        print("⚠ Needs more training")

if __name__ == "__main__":
    main()