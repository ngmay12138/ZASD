import re
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import os

# vLLM相关导入
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
import os
# 指定 SAM3 库的根目录
sam3_root = '/root/ch_trl/sam3-main'
if os.path.exists(sam3_root) and sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)
# SAM3 相关导入（可选）
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    print(f"警告：未找到SAM3库，将无法使用自检功能。详细错误：{e}")


class EnhancedRewardSystem:
    def __init__(self,
                 weights: Dict[str, float] = None,
                 judge_model_path: str = None,
                 tensor_parallel_size: int = 1,
                 use_sam3: bool = True,
                 **kwargs):
        self.default_weights = {
            'description': 0.3,
            'think': 0.3,
            'answer': 0.3,
            'format': 0.1,
        }
        self.weights = weights or self.default_weights.copy()
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # 初始化LLM考官（用于think和answer）
        self.llm_model_path = judge_model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._init_llm_judge()

        # 初始化SAM3
        self.use_sam3 = use_sam3 and SAM3_AVAILABLE
        if self.use_sam3:
            self._init_sam3()
        else:
            print("SAM3自检未启用，描述部分将使用LLM评估。")

    def _init_llm_judge(self):
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path, trust_remote_code=True
        )
        # 初始化vLLM
        self.llm = LLM(
            model=self.llm_model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.2,
            max_model_len=2048,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        print(f"LLM考官初始化完成，模型路径: {self.llm_model_path}")

    def _init_sam3(self):
        """初始化SAM3模型"""
        try:
            self.sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(self.sam3_model)
            print("SAM3模型初始化完成。")
        except Exception as e:
            print(f"SAM3初始化失败: {e}，将禁用SAM3。")
            self.use_sam3 = False

    def _extract_tag(self, text: str, tag: str) -> str:
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _build_prompt(self, gt: str, pred: str) -> str:
        template = (
            "You are an industrial anomaly detection expert. Please determine, based on the provided 'standard answer,' whether the 'predicted answer' is semantically the same as the 'standard answer.' If they are the same, output 1; otherwise, output 0.\n"
            "Standard answer：{gt}\n"
            "Predicted answer：{pred}\n\n"
            "Output："
        )
        return template.format(gt=gt, pred=pred)

    def _llm_evaluate(self, prompts: List[str]) -> List[int]:
        if not prompts:
            return []
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores = []
        for output in outputs:
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                match = re.search(r'[01]', response)
                score = int(match.group()) if match else 0
            else:
                score = 0
            scores.append(score)
        return scores

    def sam3_segment(self, image: Image.Image, text_prompt: str) -> Tuple[np.ndarray, List[float]]:
        """使用SAM3进行分割，返回掩码和边界框"""
        inference_state = self.sam3_processor.set_image(image)
        output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        if len(masks) == 0:
            return np.zeros((image.height, image.width), dtype=bool), [0, 0, 0, 0]

        # 取最高置信度的掩码
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)
        best_idx = np.argmax(scores_np)

        best_mask = masks[best_idx]
        if isinstance(best_mask, torch.Tensor):
            best_mask = best_mask.cpu().numpy()
        best_mask = best_mask > 0

        best_box = boxes[best_idx]
        if isinstance(best_box, torch.Tensor):
            best_box = best_box.cpu().tolist()
        elif isinstance(best_box, np.ndarray):
            best_box = best_box.tolist()

        return best_mask, best_box

    def compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """计算两个二值掩码的IoU"""
        pred = pred_mask > 0
        gt = gt_mask > 0
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def compute_description_reward_sam3(self, original_image_path: str, description: str, mask_path: str) -> float:
        """使用SAM3计算描述奖励（IoU）"""
        if not self.use_sam3 or not os.path.exists(mask_path):
            return 0.0
        try:
            # 加载原图和真实掩码
            image = Image.open(original_image_path).convert("RGB")
            gt_mask_img = Image.open(mask_path).convert("L")
            gt_mask = np.array(gt_mask_img) > 0

            # 使用描述作为文本提示进行SAM3分割
            pred_mask, _ = self.sam3_segment(image, description)
            iou = self.compute_iou(pred_mask, gt_mask)
            return iou
        except Exception as e:
            print(f"SAM3评估出错: {e}")
            return 0.0

    def compute_comprehensive_reward(self,
                                     completions: List[List[Dict]],
                                     solution: List[str],
                                     **kwargs) -> List[float]:
        """
        计算综合奖励。
        额外关键字参数：
            image_paths: List[str]  图像路径列表
            mask_paths: List[str]   掩码路径列表（若无则为空字符串）
        """
        contents = [completion[0]["content"] for completion in completions]
        image_paths = kwargs.get("image_paths", [None] * len(contents))
        mask_paths = kwargs.get("mask_paths", [None] * len(contents))

        # 收集需要LLM评估的think和answer部分
        all_prompts = []
        sample_indices = []   # (idx, comp_type) comp_type: 1=think, 2=answer
        for idx, (content, sol) in enumerate(zip(contents, solution)):
            gt_think = self._extract_tag(sol, 'think')
            gt_answer = self._extract_tag(sol, 'answer')
            model_think = self._extract_tag(content, 'think')
            model_answer = self._extract_tag(content, 'answer')

            # 构建think和answer的评估prompt
            prompts = [
                self._build_prompt(gt_think, model_think),
                self._build_prompt(gt_answer, model_answer)
            ]
            all_prompts.extend(prompts)
            sample_indices.append((idx, 1))  # think
            sample_indices.append((idx, 2))  # answer

        # 执行LLM评估（think和answer）
        scores = self._llm_evaluate(all_prompts)
        sample_scores = {idx: {'think': 0.0, 'answer': 0.0} for idx in range(len(contents))}
        for (idx, comp_idx), score in zip(sample_indices, scores):
            if comp_idx == 1:
                sample_scores[idx]['think'] = score
            else:
                sample_scores[idx]['answer'] = score

        # 计算每个样本的最终奖励
        rewards = []
        for idx, (content, sol) in enumerate(zip(contents, solution)):
            # 获取think和answer奖励
            think_reward = sample_scores[idx]['think']
            answer_reward = sample_scores[idx]['answer']

            # 处理description奖励：优先使用SAM3（仅当有掩码时）
            model_desc = self._extract_tag(content, 'description')
            mask_path = mask_paths[idx] if idx < len(mask_paths) else None
            image_path = image_paths[idx] if idx < len(image_paths) else None

            if self.use_sam3 and mask_path and os.path.exists(mask_path) and image_path:
                # 使用SAM3计算IoU作为描述奖励
                desc_reward = self.compute_description_reward_sam3(image_path, model_desc, mask_path)
            else:
                # 回退：使用LLM评估描述（与think相同方式）
                gt_desc = self._extract_tag(sol, 'description')
                desc_prompt = self._build_prompt(gt_desc, model_desc)
                desc_score = self._llm_evaluate([desc_prompt])[0]
                desc_reward = float(desc_score)

            # 格式奖励
            format_reward = self._format_reward(content, sol)

            components = {
                'description': desc_reward,
                'think': think_reward,
                'answer': answer_reward,
                'format': format_reward,
            }
            total_reward = sum(components[comp] * self.weights.get(comp, 0.0)
                               for comp in components)

            if kwargs.get('normalize', False):
                total_reward = self._normalize_reward(total_reward)
            base_reward = kwargs.get('base_reward', 0.0)
            total_reward = max(total_reward, base_reward)

            rewards.append(total_reward)

        return rewards

    def _format_reward(self, content: str, solution: str) -> float:
        score = 0.0
        if re.search(r'<answer>.*?</answer>', content, re.DOTALL):
            score += 0.3
        if re.search(r'<description>.*?</description>', content, re.DOTALL):
            score += 0.3
        if re.search(r'<think>.*?</think>', content, re.DOTALL):
            score += 0.3
        if not re.search(r'\n\s*\n\s*\n', content):
            score += 0.1
        return min(score, 1.0)

    def _normalize_reward(self, reward: float) -> float:
        return 1 / (1 + math.exp(-8 * (reward - 0.5)))