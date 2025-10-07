# main_fast.py
import torch
import numpy as np
import yaml
import time
from pathlib import Path

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from training.fast_trainer import FastPlacementTrainer

def main():
    print("="*60)
    print("FAST DQC TRAINING")
    print("="*60)
    
    start_time = time.time()
    
    # 设置种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载快速配置
    with open('configs/training_config_fast.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建问题和环境
    problem_config = config['problem']['small']
    generator = QuantumProblemGenerator(**problem_config, seed=42)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    env = DQCEnvironment(problem_data, {'max_steps': 200})
    
    # 创建网络
    model_config = config['model']
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # 创建PPO
    ppo = PPO(policy_net, value_net, config['ppo'])
    
    # 创建快速训练器
    trainer = FastPlacementTrainer(env, policy_net, value_net, ppo, config)
    
    # Stage 1: 快速placement训练（目标：5分钟内）
    print("\nStage 1: Fast Placement Training")
    trainer.fast_placement_training(num_epochs=30)
    
    # Stage 2: 快速集成训练（目标：10分钟内）
    print("\nStage 2: Integrated Training")
    trainer.integrated_training(total_timesteps=5000)
    
    # 保存模型
    torch.save({
        'policy_state': policy_net.state_dict(),
        'value_state': value_net.state_dict(),
        'config': config
    }, 'fast_model.pt')
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes!")
    
    # 快速测试
    print("\nQuick Test:")
    env.reset()
    total_reward = 0
    
    for _ in range(32):
        state_repr = env.state.get_state_representation()
        valid_actions = env.get_valid_actions()
        
        if not valid_actions['map'] and not valid_actions['schedule']:
            break
        
        with torch.no_grad():
            action, _ = policy_net.get_action(state_repr, valid_actions, deterministic=True)
        
        _, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Test reward: {total_reward:.2f}")
    print(f"Completion: {env._get_completion_rate():.2%}")

if __name__ == "__main__":
    main()