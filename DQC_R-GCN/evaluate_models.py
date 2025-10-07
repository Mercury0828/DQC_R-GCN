# evaluate_models_fixed.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from expert_policy.uniq_expert import UNIQExpertPolicy

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, problem_config: Dict, num_episodes: int = 10):
        self.problem_config = problem_config
        self.num_episodes = num_episodes
        self.results = {}
        
    def evaluate_model(self, env, policy_func, model_name: str):
        """评估单个模型"""
        print(f"\nEvaluating {model_name}...")
        
        episodes_data = []
        
        for episode in range(self.num_episodes):
            env.reset()
            episode_info = {
                'rewards': [],
                'actions': [],
                'mappings': {},
            }
            
            for step in range(200):
                state_repr = env.state.get_state_representation()
                valid_actions = env.get_valid_actions()
                
                if not valid_actions['map'] and not valid_actions['schedule']:
                    break
                
                action = policy_func(state_repr, valid_actions)
                
                if action[0] == 'map':
                    qubit, qpu = action[1]
                    episode_info['mappings'][qubit] = qpu
                
                _, reward, done, _ = env.step(action)
                
                episode_info['rewards'].append(reward)
                episode_info['actions'].append(action)
                
                if done:
                    break
            
            # 计算指标
            episode_info['total_reward'] = sum(episode_info['rewards'])
            episode_info['completion_rate'] = env._get_completion_rate()
            episode_info['final_comm_cost'] = self._calculate_comm_cost(env)
            episode_info['num_steps'] = len(episode_info['actions'])
            episode_info['local_gates'], episode_info['remote_gates'] = self._count_gate_types(env)
            
            episodes_data.append(episode_info)
            
            if episode % 3 == 0 or episode == 0:
                print(f"  Episode {episode+1}/{self.num_episodes}: "
                      f"Reward={episode_info['total_reward']:.2f}, "
                      f"Completion={episode_info['completion_rate']:.2%}, "
                      f"CommCost={episode_info['final_comm_cost']:.2f}")
        
        self.results[model_name] = episodes_data
        return episodes_data
    
    def _calculate_comm_cost(self, env):
        """计算通信成本"""
        total_cost = 0
        for g_id, gate in env.problem_data['gates'].items():
            control = gate['control']
            target = gate['target']
            
            if control in env.state.qubit_mapping and target in env.state.qubit_mapping:
                c_qpu = env.state.qubit_mapping[control]
                t_qpu = env.state.qubit_mapping[target]
                
                if c_qpu != t_qpu:
                    cost = env.problem_data['C'].get((c_qpu, t_qpu), 0)
                    total_cost += cost
        
        return total_cost
    
    def _count_gate_types(self, env):
        """统计gate类型"""
        local = 0
        remote = 0
        
        for g_id, gate in env.problem_data['gates'].items():
            control = gate['control']
            target = gate['target']
            
            if control in env.state.qubit_mapping and target in env.state.qubit_mapping:
                c_qpu = env.state.qubit_mapping[control]
                t_qpu = env.state.qubit_mapping[target]
                
                if c_qpu == t_qpu:
                    local += 1
                else:
                    remote += 1
        
        return local, remote
    
    def print_comparison(self):
        """打印详细对比结果"""
        print("\n" + "="*80)
        print(" MODEL COMPARISON RESULTS")
        print("="*80)
        
        # 表头
        print(f"\n{'Model':<20} {'Avg Reward':<15} {'Completion':<12} {'Comm Cost':<15} "
              f"{'Local/Remote':<15} {'Steps':<8}")
        print("-"*80)
        
        # 数据
        for model_name, episodes in self.results.items():
            rewards = [ep['total_reward'] for ep in episodes]
            completions = [ep['completion_rate'] for ep in episodes]
            comm_costs = [ep['final_comm_cost'] for ep in episodes]
            steps = [ep['num_steps'] for ep in episodes]
            local_gates = [ep['local_gates'] for ep in episodes]
            remote_gates = [ep['remote_gates'] for ep in episodes]
            
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            avg_completion = np.mean(completions)
            avg_comm = np.mean(comm_costs)
            std_comm = np.std(comm_costs)
            avg_local = np.mean(local_gates)
            avg_remote = np.mean(remote_gates)
            avg_steps = np.mean(steps)
            
            print(f"{model_name:<20} {avg_reward:>7.2f}±{std_reward:<5.2f} "
                  f"{avg_completion:>10.2%}  {avg_comm:>7.2f}±{std_comm:<5.2f} "
                  f"{avg_local:>5.1f}/{avg_remote:<5.1f}  {avg_steps:>6.1f}")
        
        print("="*80)
        
        # 性能分析
        self._analyze_performance()
        
        # 生成图表
        self._plot_results()
    
    def _analyze_performance(self):
        """分析性能差距"""
        print("\n" + "="*80)
        print(" PERFORMANCE ANALYSIS")
        print("="*80)
        
        if 'Expert' not in self.results:
            print("No Expert baseline found!")
            return
        
        expert_data = self.results['Expert']
        expert_comm = np.mean([ep['final_comm_cost'] for ep in expert_data])
        expert_reward = np.mean([ep['total_reward'] for ep in expert_data])
        expert_local = np.mean([ep['local_gates'] for ep in expert_data])
        expert_remote = np.mean([ep['remote_gates'] for ep in expert_data])
        
        print(f"\nExpert Baseline:")
        print(f"  Communication cost: {expert_comm:.2f}")
        print(f"  Total reward: {expert_reward:.2f}")
        print(f"  Local/Remote gates: {expert_local:.0f}/{expert_remote:.0f}")
        
        for model_name, episodes in self.results.items():
            if model_name == 'Expert':
                continue
            
            model_comm = np.mean([ep['final_comm_cost'] for ep in episodes])
            model_reward = np.mean([ep['total_reward'] for ep in episodes])
            model_local = np.mean([ep['local_gates'] for ep in episodes])
            model_remote = np.mean([ep['remote_gates'] for ep in episodes])
            
            comm_ratio = model_comm / expert_comm if expert_comm > 0 else 0
            reward_diff = model_reward - expert_reward
            
            print(f"\n{model_name} Performance:")
            print(f"  Communication cost: {model_comm:.2f} ({comm_ratio:.2f}x of Expert)")
            print(f"  Reward difference: {reward_diff:+.2f}")
            print(f"  Local/Remote gates: {model_local:.0f}/{model_remote:.0f}")
            
            # 评级
            print(f"\n  Quality Assessment:")
            if comm_ratio < 1.2:
                print(f"    ✅ EXCELLENT - Within 20% of Expert")
            elif comm_ratio < 1.5:
                print(f"    ✓ GOOD - Acceptable performance")
            elif comm_ratio < 2.0:
                print(f"    ⚠️ POOR - Needs improvement")
            else:
                print(f"    ❌ VERY POOR - Major issues with placement strategy")
                print(f"    → Communication cost is {comm_ratio:.1f}x higher than Expert!")
                print(f"    → Model is not learning optimal qubit placement")
                
            # 建议
            if comm_ratio > 1.5:
                print(f"\n  Recommendations:")
                print(f"    1. Increase r_colocate reward (currently too low)")
                print(f"    2. Increase r_separate penalty (currently too low)")
                print(f"    3. More BC pretraining focused on placement")
                print(f"    4. Consider curriculum learning: placement first, then scheduling")
    
    def _plot_results(self):
        """生成对比图表"""
        if len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        model_names = list(self.results.keys())
        colors = ['green' if m == 'Expert' else 'orange' for m in model_names]
        
        # 1. Communication Cost
        ax = axes[0]
        comm_means = []
        comm_stds = []
        for model in model_names:
            costs = [ep['final_comm_cost'] for ep in self.results[model]]
            comm_means.append(np.mean(costs))
            comm_stds.append(np.std(costs))
        
        bars = ax.bar(range(len(model_names)), comm_means, color=colors, alpha=0.7)
        ax.errorbar(range(len(model_names)), comm_means, yerr=comm_stds,
                   fmt='none', color='black', capsize=5)
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, comm_means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean:.1f}', ha='center', va='bottom')
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        ax.set_title('Communication Cost (Lower is Better)')
        ax.set_ylabel('Cost')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加Expert基准线
        if 'Expert' in self.results:
            expert_comm = comm_means[model_names.index('Expert')]
            ax.axhline(y=expert_comm, color='green', linestyle='--', alpha=0.5, label='Expert Level')
            ax.legend()
        
        # 2. Total Reward
        ax = axes[1]
        reward_means = []
        reward_stds = []
        for model in model_names:
            rewards = [ep['total_reward'] for ep in self.results[model]]
            reward_means.append(np.mean(rewards))
            reward_stds.append(np.std(rewards))
        
        bars = ax.bar(range(len(model_names)), reward_means, color=colors, alpha=0.7)
        ax.errorbar(range(len(model_names)), reward_means, yerr=reward_stds,
                   fmt='none', color='black', capsize=5)
        
        for i, (bar, mean) in enumerate(zip(bars, reward_means)):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 1 if mean > 0 else bar.get_height() - 3,
                   f'{mean:.1f}', ha='center', va='bottom' if mean > 0 else 'top')
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        ax.set_title('Total Reward (Higher is Better)')
        ax.set_ylabel('Reward')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Gate Locality
        ax = axes[2]
        width = 0.35
        x = np.arange(len(model_names))
        
        local_means = []
        remote_means = []
        for model in model_names:
            local = [ep['local_gates'] for ep in self.results[model]]
            remote = [ep['remote_gates'] for ep in self.results[model]]
            local_means.append(np.mean(local))
            remote_means.append(np.mean(remote))
        
        bars1 = ax.bar(x - width/2, local_means, width, label='Local Gates', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, remote_means, width, label='Remote Gates', color='red', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_title('Gate Locality (More Local is Better)')
        ax.set_ylabel('Number of Gates')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Comparison plot saved to 'model_comparison.png'")


def load_model(model_path: str, env):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_config = checkpoint.get('config', {}).get('model', {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_bases': 8,
        'dropout': 0.1,
        'num_heads': 4
    })
    
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    
    # 初始化
    dummy_state = env.reset()
    dummy_state_repr = env.state.get_state_representation()
    dummy_valid_actions = env.get_valid_actions()
    with torch.no_grad():
        try:
            _ = policy_net.forward(dummy_state_repr, dummy_valid_actions)
        except:
            pass
    
    policy_net.load_state_dict(checkpoint['policy_state'], strict=False)
    policy_net.eval()
    
    return policy_net


def main():
    print("="*80)
    print(" DQC MODEL EVALUATION")
    print("="*80)
    
    # 配置
    problem_config = {
        'num_gates': 20,
        'num_qpus': 3,
        'num_qubits': 12,
        'qpu_capacity': 5,
        'time_horizon': 30,
    }
    
    # 创建环境
    generator = QuantumProblemGenerator(**problem_config, seed=12222)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    env = DQCEnvironment(problem_data, {'max_steps': 200})
    
    # 创建评估器
    evaluator = ModelEvaluator(problem_config, num_episodes=10)
    
    # 1. 评估Expert
    expert = UNIQExpertPolicy(env, debug=False)
    evaluator.evaluate_model(
        env, 
        lambda s, v: expert.get_expert_action(s, v),
        "Expert"
    )
    
    # 2. 评估模型
    model_files = [
        'fast_model.pt'
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"\nFound model: {model_file}")
            model = load_model(model_file, env)
            
            def model_policy(state_repr, valid_actions):
                with torch.no_grad():
                    action, _ = model.get_action(state_repr, valid_actions, deterministic=True)
                return action
            
            model_name = model_file.replace('.pt', '').replace('_', ' ').title()
            evaluator.evaluate_model(env, model_policy, model_name)
            break
    
    # 生成报告
    evaluator.print_comparison()


if __name__ == "__main__":
    main()