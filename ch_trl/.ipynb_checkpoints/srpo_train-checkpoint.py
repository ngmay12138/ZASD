from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from trl.rewards import accuracy_reward
from reward_sim import EnhancedRewardSystem
import json
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from pathlib import Path

class RewardLoggerCallback(TrainerCallback):
    """自定义回调函数记录奖励值"""
    
    def __init__(self, save_dir, save_freq):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_freq = save_freq
        self.rewards = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在日志时记录奖励值"""
        if logs is not None:
            # GRPO中奖励值通常以'reward'或'mean_reward'等字段记录
            reward_keys = ['reward', 'mean_reward', 'rewards/mean', 
                          'rewards', 'avg_reward']
            
            for key in reward_keys:
                if key in logs:
                    step = state.global_step
                    reward = logs[key]
                    
                    self.rewards.append(reward)
                    self.steps.append(step)
                    
                    # 定期保存数据
                    if step % self.save_freq == 0:
                        self.save_data()
                        self.plot_rewards()
                    break
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时保存最终结果"""
        self.save_data()
        self.plot_rewards()
        
    def save_data(self):
        """保存奖励数据到JSON文件"""
        data = {
            'steps': self.steps,
            'rewards': self.rewards
        }
        
        save_path = self.save_dir / f"rewards_step_{self.steps[-1] if self.steps else 0}.json"
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 同时保存为CSV
        csv_path = self.save_dir / "rewards_history.csv"
        import pandas as pd
        df = pd.DataFrame({'step': self.steps, 'reward': self.rewards})
        df.to_csv(csv_path, index=False)
    
    def plot_rewards(self):
        """绘制奖励曲线"""
        if len(self.steps) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.rewards, 'b-', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Average Reward')
            plt.title('GRPO Training Reward Curve')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            plot_path = self.save_dir / f"reward_curve_step_{self.steps[-1]}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
if __name__ == '__main__':
    # 1. 加载数据集
    dataset = load_dataset('json', data_files="t1.0_3k_grpo_train.json", split="all")
    print("数据集第一个样本的键：", dataset[0].keys())  # 查看实际列名

    # 2. 配置训练参数
    training_args = GRPOConfig(
        output_dir="my_model/qwen3_vl_2b_srpo",
        num_train_epochs=3,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        logging_steps=10,
        use_vllm=True,
        vllm_mode="colocate",
        # vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=4096,
        save_steps=10000,
    )

    # 3. 初始化奖励记录器
    reward_logger = RewardLoggerCallback(
        save_dir="grpo_reward_logs",
        save_freq=1000
    )

    # 4. 初始化自定义奖励系统（注意 judge_model_path 路径需存在）
    custom_reward_system = EnhancedRewardSystem(
        judge_model_path="Qwen3-VL-2B",
        tensor_parallel_size=4
    )

    def custom_reward_func(completions, solution, **kwargs):
        return custom_reward_system.compute_comprehensive_reward(
            completions, solution,
            normalize=False,
            base_reward=0.01
        )

    # 5. 创建 Trainer
    trainer = GRPOTrainer(
        model="qwen3_2b_20_5k",
        args=training_args,
        train_dataset=dataset,
        reward_funcs=custom_reward_func,
        callbacks=[reward_logger],
    )

    # 6. 开始训练
    trainer.train()