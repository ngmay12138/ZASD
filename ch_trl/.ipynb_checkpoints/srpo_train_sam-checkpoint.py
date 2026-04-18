from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from trl.rewards import accuracy_reward
from reward_sam import EnhancedRewardSystem
import json
import math
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
        if logs is not None:
            reward_keys = ['reward', 'mean_reward', 'rewards/mean',
                           'rewards', 'avg_reward']
            for key in reward_keys:
                if key in logs:
                    step = state.global_step
                    reward = logs[key]
                    self.rewards.append(reward)
                    self.steps.append(step)
                    if step % self.save_freq == 0:
                        self.save_data()
                        self.plot_rewards()
                    break

    def on_train_end(self, args, state, control, **kwargs):
        self.save_data()
        self.plot_rewards()

    def save_data(self):
        data = {
            'steps': self.steps,
            'rewards': self.rewards
        }
        save_path = self.save_dir / f"rewards_step_{self.steps[-1] if self.steps else 0}.json"
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        import pandas as pd
        csv_path = self.save_dir / "rewards_history.csv"
        df = pd.DataFrame({'step': self.steps, 'reward': self.rewards})
        df.to_csv(csv_path, index=False)

    def plot_rewards(self):
        if len(self.steps) > 1:
            import numpy as np
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.rewards, color='#5B9BD5', linewidth=1, alpha=0.4, label='Raw Reward')
            # 平滑奖励（TensorBoard风格EMA，sigmoid自适应权重）
            weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(self.rewards))) - 0.5)
            smoothed = []
            last = self.rewards[0]
            for val in self.rewards:
                last = last * weight + (1 - weight) * val
                smoothed.append(last)
            plt.plot(self.steps, smoothed, color='#1F5FAD', linewidth=2, label='Smoothed Reward')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Reward')
            plt.title('GRPO Training Reward Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = self.save_dir / f"reward_curve_step_{self.steps[-1]}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    # 1. 加载数据集并提取 image_path / mask_path
    dataset = load_dataset('json', data_files="sam4grpo_train.json", split="all")

    def extract_paths(example):
        for msg in example['prompt']:
            if msg['role'] == 'user':
                for c in msg['content']:
                    if c.get('type') == 'image':
                        img = c['image']
                        example['image_path'] = img
                        import os
                        example['mask_path'] = os.path.splitext(img)[0] + '.png'
                        return example
        example['image_path'] = None
        example['mask_path'] = None
        return example

    dataset = dataset.map(extract_paths)
    print("数据集第一个样本的键：", dataset[0].keys())

    # 2. 配置训练参数
    training_args = GRPOConfig(
        output_dir="my_model/qwen3_samsftrl",
        num_train_epochs=1,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        logging_steps=10,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_max_model_length=4096,
        vllm_importance_sampling_correction=False,
        save_steps=5000,
        beta=0.04,
        loss_type="grpo",
        max_completion_length=1024,
        temperature=1.0,
    )

    # 3. 初始化奖励记录器
    reward_logger = RewardLoggerCallback(
        save_dir="grpo_reward_logs",
        save_freq=1000
    )

    # 4. 初始化自定义奖励系统（启用SAM3）
    custom_reward_system = EnhancedRewardSystem(
        judge_model_path="my_model/judger_0.6b",
        tensor_parallel_size=4,
        use_sam3=True,                # 启用SAM3
    )

    def custom_reward_func(completions, solution, **kwargs):
        # 从 kwargs 中提取图像和掩码路径（需在数据集中提供）
        image_paths = kwargs.get("image_path", [None] * len(completions))
        mask_paths = kwargs.get("mask_path", [None] * len(completions))
        return custom_reward_system.compute_comprehensive_reward(
            completions, solution,
            normalize=False,
            base_reward=0.0,
            image_paths=image_paths,
            mask_paths=mask_paths
        )

    # 5. 创建 Trainer
    trainer = GRPOTrainer(
        model="qwen3_sam4sft/checkpoint-500",
        args=training_args,
        train_dataset=dataset,
        reward_funcs=custom_reward_func,
        callbacks=[reward_logger],
    )

    # 6. 开始训练
    trainer.train()