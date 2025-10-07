import torch
import numpy as np
from pathlib import Path
import yaml

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from expert_policy.uniq_expert import UNIQExpertPolicy

class HybridPolicy:
    """混合策略：在必要时使用专家指导"""
    
    def __init__(self, policy_net, expert, force_schedule_ratio=0.3):
        self.policy_net = policy_net
        self.expert = expert
        self.force_schedule_ratio = force_schedule_ratio
        self.map_count = 0
        self.schedule_count = 0
    
    def get_action(self, state_repr, valid_actions):
        """获取动作，必要时强制调度"""
        
        # 统计可用动作
        num_map = len(valid_actions.get('map', []))
        num_schedule = len(valid_actions.get('schedule', []))
        
        # 如果只有schedule动作可用，直接用模型
        if num_schedule > 0 and num_map == 0:
            with torch.no_grad():
                action, _ = self.policy_net.get_action(state_repr, valid_actions, deterministic=True)
            self.schedule_count += 1
            return action
        
        # 如果map和schedule都可用
        if num_schedule > 0 and num_map > 0:
            # 如果map太多了，强制选择schedule
            if self.map_count > 0 and self.map_count / (self.map_count + self.schedule_count + 1) > 0.7:
                # 从可用的schedule动作中选择
                schedule_actions = valid_actions['schedule']
                if schedule_actions:
                    # 让专家选择最好的schedule动作
                    expert_action = self.expert.get_expert_action(state_repr, {
                        'map': [],
                        'schedule': schedule_actions
                    })
                    self.schedule_count += 1
                    return expert_action
        
        # 默认：使用训练的模型
        with torch.no_grad():
            action, _ = self.policy_net.get_action(state_repr, valid_actions, deterministic=True)
        
        if action[0] == 'map':
            self.map_count += 1
        else:
            self.schedule_count += 1
        
        return action

def evaluate_with_hybrid(policy_net, env, num_episodes=10):
    """使用混合策略评估"""
    expert = UNIQExpertPolicy(env, debug=False)
    hybrid = HybridPolicy(policy_net, expert)
    
    results = []
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        hybrid.map_count = 0
        hybrid.schedule_count = 0
        
        for _ in range(500):
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            action = hybrid.get_action(state_repr, valid_actions)
            _, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        completion = env._get_completion_rate()
        results.append({
            'reward': total_reward,
            'completion': completion,
            'steps': steps,
            'map_actions': hybrid.map_count,
            'schedule_actions': hybrid.schedule_count
        })
        
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
              f"Completion={completion:.2%}, Steps={steps}, "
              f"Actions: map={hybrid.map_count}, schedule={hybrid.schedule_count}")
    
    return results

def evaluate_with_exploration(policy_net, env, num_episodes=10):
    """使用探索性评估（非确定性）"""
    results = []
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        action_counts = {'map': 0, 'schedule': 0}
        
        for _ in range(500):
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            # 使用随机性
            with torch.no_grad():
                action, _ = policy_net.get_action(
                    state_repr, valid_actions, 
                    deterministic=False  # 使用随机性
                )
            
            _, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            action_counts[action[0]] += 1
            
            if done:
                break
        
        completion = env._get_completion_rate()
        results.append({
            'reward': total_reward,
            'completion': completion,
            'steps': steps,
            'map_actions': action_counts['map'],
            'schedule_actions': action_counts['schedule']
        })
        
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
              f"Completion={completion:.2%}, Steps={steps}, "
              f"Actions: map={action_counts['map']}, schedule={action_counts['schedule']}")
    
    return results

def main():
    # 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    problem_config = config['problem']['small']
    model_config = config['model']
    
    # 生成问题
    generator = QuantumProblemGenerator(**problem_config, seed=42)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    env = DQCEnvironment(problem_data, {'max_steps': 200})
    
    # 创建和加载模型
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # 初始化
    env.reset()
    state_repr = env.state.get_state_representation()
    _ = policy_net.rgcn(state_repr)
    
    # 加载最佳模型
    if Path('best_model.pt').exists():
        checkpoint = torch.load('best_model.pt', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_state'])
        policy_net.eval()
        
        print("="*60)
        print("评估1：混合策略（模型+专家纠正）")
        print("="*60)
        hybrid_results = evaluate_with_hybrid(policy_net, env, num_episodes=10)
        
        print("\n" + "="*60)
        print("评估2：随机探索策略")
        print("="*60)
        explore_results = evaluate_with_exploration(policy_net, env, num_episodes=10)
        
        # 统计
        print("\n" + "="*60)
        print("总结")
        print("="*60)
        
        if hybrid_results:
            avg_reward = np.mean([r['reward'] for r in hybrid_results])
            avg_completion = np.mean([r['completion'] for r in hybrid_results])
            print(f"混合策略: 平均奖励={avg_reward:.2f}, 平均完成率={avg_completion:.2%}")
        
        if explore_results:
            avg_reward = np.mean([r['reward'] for r in explore_results])
            avg_completion = np.mean([r['completion'] for r in explore_results])
            print(f"探索策略: 平均奖励={avg_reward:.2f}, 平均完成率={avg_completion:.2%}")

if __name__ == "__main__":
    main()