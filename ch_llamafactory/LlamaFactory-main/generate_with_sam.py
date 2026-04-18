import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional, Tuple
import argparse
import random
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# vLLM相关导入
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
import os

# 指定 SAM3 库的根目录
sam3_root = '/root/ch_llamafactory/sam3-main'
if os.path.exists(sam3_root) and sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)
# SAM3相关导入（需提前安装）
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    print(f"警告：未找到SAM3库，将无法使用自检功能。详细错误：{e}")


class StructuredOutputGenerator:
    def __init__(self, model_path, device=None, batch_size=4, tensor_parallel_size=1,
                 use_sam3=True, sam3_iou_threshold=0.5):
        """
        初始化结构化输出生成器

        Args:
            model_path: Qwen3-VL模型路径
            device: 使用的设备，默认为cuda
            batch_size: 批处理大小
            tensor_parallel_size: GPU并行数量
            use_sam3: 是否启用SAM3自检（仅对异常样本）
            sam3_iou_threshold: SAM3预测掩码与真实掩码的IoU阈值，高于此值认为一致
        """
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sam3 = use_sam3 and SAM3_AVAILABLE
        self.sam3_iou_threshold = sam3_iou_threshold

        print(f"使用设备: {self.device}")
        print(f"批处理大小: {batch_size}")
        print(f"SAM3自检: {'启用' if self.use_sam3 else '禁用'}")

        # 定义正常和异常图像的提示词
        self.normal_prompt = """
## Task Description
You are an industrial anomaly detection expert and need to explain why there is no abnormal situation in the objects in the image.
## Output Format Requirements
Answer strictly in the following format. Replace the think with your response, and do not add any additional text:
<think>Briefly describe your thought process about why the object in the image is normal in 1-2 sentences</think><answer>normal</answer>
Now analyze this image:"""

        # 异常图像提示词：结合原图和掩码图
        self.abnormal_prompt = """
## Task Description
You are an industrial anomaly detection expert. You are given two images: the first is the original object image, and the second is a defect mask image (highlighting the anomalous regions). Identify the defects, damage, abnormal states, or non-compliance issues in the object based on both images.
## Output Format Requirements
Answer strictly in the following format. Replace the description and think with your response, and do not add any additional text:
<description>A short phrase (3-8 words) describing the anomaly's appearance and location</description><think>Briefly describe your thought process about why the object in the image is abnormal in 1-2 sentences</think><answer>abnormal</answer>
Now analyze the images:"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 初始化vLLM
        print("正在初始化vLLM引擎...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16" if self.device == "cuda" else "float32",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.42,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},
        )
        print("vLLM引擎初始化完成！")

        # 初始化SAM3模型（如果启用）
        if self.use_sam3:
            print("正在初始化SAM3模型...")
            self.sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(self.sam3_model)
            print("SAM3模型初始化完成！")

    def get_sampling_params(self, temperature=0.0):
        """获取不同temperature的采样参数"""
        return SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=1000,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

    def build_prompt(self, is_normal: bool, has_mask: bool = False) -> str:
        """根据图像类型构建提示词（文本部分）"""
        if is_normal:
            prompt = self.normal_prompt
        else:
            prompt = self.abnormal_prompt

        if is_normal:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_text

    def parse_structured_output(self, response: str, is_normal: bool) -> Dict:
        """解析结构化输出，只提取特定格式标签内的内容。

        使用 rfind 从后往前匹配，兼容 Qwen3-VL 原生思考模式：
        模型生成的 response 可能以 CoT 推理内容开头（因 chat template 在 prompt 末尾
        注入了 <think> 触发词），结构如下：
            [Qwen3 内部推理]</think>
            <think>结构化think内容</think><answer>normal/abnormal</answer>
        原始 find() 会将原生推理的 </think> 当作结构化 think 的结束标签，
        导致 think_end < think_start，字段永远为空。
        改用 rfind() 从末尾向前查找，确保拿到最后一组完整标签对。
        """
        result = {
            "raw_response": response,
            "description": "",
            "location": "",   # 仅异常样本可能有
            "think": "",
            "answer": "",
            "parsing_success": False,
            "parsing_error": None
        }

        try:
            # 1. 先定位最后一个 <answer>...</answer>
            answer_end_tag = response.rfind('</answer>')
            answer_start_tag = response.rfind('<answer>', 0, answer_end_tag)
            if answer_start_tag != -1 and answer_end_tag > answer_start_tag:
                result["answer"] = response[answer_start_tag + len('<answer>'):answer_end_tag].strip()

            # 2. 在 <answer> 之前区域找最后一个 <think>...</think>
            pre_answer = response[:answer_start_tag] if answer_start_tag != -1 else response
            think_end_tag = pre_answer.rfind('</think>')
            think_start_tag = pre_answer.rfind('<think>', 0, think_end_tag)
            if think_start_tag != -1 and think_end_tag > think_start_tag:
                result["think"] = pre_answer[think_start_tag + len('<think>'):think_end_tag].strip()

            # 3. 在 <think> 之前区域找最后一个 <description>...</description>
            pre_think = pre_answer[:think_start_tag] if think_start_tag != -1 else pre_answer
            desc_end_tag = pre_think.rfind('</description>')
            desc_start_tag = pre_think.rfind('<description>', 0, desc_end_tag)
            if desc_start_tag != -1 and desc_end_tag > desc_start_tag:
                result["description"] = pre_think[desc_start_tag + len('<description>'):desc_end_tag].strip()

            # 4. 在 <think> 之前区域找最后一个 <location>...</location>（可选）
            loc_end_tag = pre_think.rfind('</location>')
            loc_start_tag = pre_think.rfind('<location>', 0, loc_end_tag)
            if loc_start_tag != -1 and loc_end_tag > loc_start_tag:
                result["location"] = pre_think[loc_start_tag + len('<location>'):loc_end_tag].strip()

            # 检查必需字段（description 为可选，正常样本 prompt 中无此字段）
            if result["think"] and result["answer"]:
                if result["answer"].lower() in ["normal", "abnormal"]:
                    result["parsing_success"] = True
                else:
                    result["parsing_error"] = f"无效的answer值: {result['answer']}"
            else:
                missing = []
                if not result["think"]: missing.append("think")
                if not result["answer"]: missing.append("answer")
                result["parsing_error"] = f"缺失字段: {', '.join(missing)}"

        except Exception as e:
            result["parsing_error"] = f"解析异常: {str(e)}"

        return result

    def sam3_segment(self, image: Image.Image, text_prompt: str) -> Tuple[np.ndarray, List[float]]:
        inference_state = self.sam3_processor.set_image(image)
        output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
    
        if len(masks) == 0:
            return np.zeros((image.height, image.width), dtype=bool), [0, 0, 0, 0]
    
        # 将 scores 转换为 NumPy 数组（如果它是 PyTorch 张量，先移到 CPU）
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)
        best_idx = np.argmax(scores_np)
    
        # 获取最佳掩码并转换为 NumPy 布尔数组
        best_mask = masks[best_idx]
        if isinstance(best_mask, torch.Tensor):
            best_mask = best_mask.cpu().numpy()
        best_mask = best_mask > 0  # 确保布尔类型
    
        # 获取最佳边界框并转换为 Python 列表
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

    def verify_with_sam3(self, original_image: Image.Image, description: str,
                         gt_mask_path: str) -> Tuple[bool, Optional[List[float]], Optional[np.ndarray], float]:
        """
        使用SAM3验证描述与真实掩码的一致性。
        返回：(是否通过, 预测边界框, 预测掩码, IoU值)
        """
        if not self.use_sam3:
            # 未启用自检，默认通过
            return True, None, None, -1.0

        # 加载真实掩码（二值）
        try:
            gt_mask_img = Image.open(gt_mask_path).convert('L')
            gt_mask = np.array(gt_mask_img) > 0
        except Exception as e:
            print(f"无法加载真实掩码 {gt_mask_path}: {e}")
            return False, None, None, -1.0

        # 用描述作为文本提示进行SAM3分割
        pred_mask, pred_bbox = self.sam3_segment(original_image, description)

        # 计算IoU
        iou = self.compute_iou(pred_mask, gt_mask)
        print(f"SAM3 IoU: {iou:.4f} (阈值: {self.sam3_iou_threshold})")

        if iou >= self.sam3_iou_threshold:
            return True, pred_bbox, pred_mask, iou
        else:
            return False, None, None, iou

    def plot_iou_statistics(self, iou_records: List[Dict], save_path: str):
        """绘制所有异常样本的IoU原始分布可视化图（单图）。"""
        if not iou_records:
            print("没有IoU记录，跳过绘图。")
            return

        iou_values = np.array([r["iou"] for r in iou_records])
        mean_iou = float(np.mean(iou_values))
        median_iou = float(np.median(iou_values))
        std_iou = float(np.std(iou_values))

        fig, (ax_hist, ax_scatter) = plt.subplots(2, 1, figsize=(12, 8),
                                                   gridspec_kw={'height_ratios': [1, 1]})
        fig.suptitle(f"SAM3 IoU Distribution  (n={len(iou_values)}, "
                     f"mean={mean_iou:.4f}, median={median_iou:.4f}, std={std_iou:.4f})",
                     fontsize=13, fontweight='bold')

        # --- 上: 直方图 ---
        bins = np.linspace(0, 1, 21)
        ax_hist.hist(iou_values, bins=bins, alpha=0.8, color='#3498db', edgecolor='white')
        ax_hist.axvline(x=mean_iou, color='#e74c3c', linestyle='--', linewidth=1.5,
                        label=f'Mean = {mean_iou:.4f}')
        ax_hist.axvline(x=median_iou, color='#f39c12', linestyle='-.', linewidth=1.5,
                        label=f'Median = {median_iou:.4f}')
        ax_hist.set_xlabel('IoU')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('IoU Histogram')
        ax_hist.legend()

        # --- 下: 逐样本散点图 ---
        ax_scatter.scatter(np.arange(len(iou_values)), iou_values,
                           s=12, alpha=0.5, color='#3498db')
        ax_scatter.axhline(y=mean_iou, color='#e74c3c', linestyle='--', linewidth=1.5,
                           label=f'Mean = {mean_iou:.4f}')
        ax_scatter.set_xlabel('Sample Index')
        ax_scatter.set_ylabel('IoU')
        ax_scatter.set_title('IoU per Sample')
        ax_scatter.set_ylim(-0.05, 1.05)
        ax_scatter.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"IoU统计图已保存至 {save_path}")

    def batch_generate_structured_output(self, image_paths: List[str], is_normal_list: List[bool],
                                         mask_paths: List[str] = None, temperature=0.0,
                                         iou_records: List[Dict] = None) -> List[Dict]:
        """
        批量生成结构化输出，并对异常样本进行SAM3自检。
        返回的结果中，如果自检不通过则不会包含该条目。
        iou_records: 可选列表，用于收集每个异常样本的IoU记录。
        """
        if not image_paths:
            return []

        if len(image_paths) != len(is_normal_list):
            raise ValueError(f"长度不匹配: {len(image_paths)} vs {len(is_normal_list)}")

        if mask_paths is not None and len(mask_paths) != len(image_paths):
            raise ValueError(f"mask_paths长度不匹配: {len(image_paths)} vs {len(mask_paths)}")

        batch_inputs = []
        valid_indices = []  # 保存原索引与是否使用SAM3验证的额外信息

        # 第一遍：构建vLLM输入
        for i, (img_path, is_normal) in enumerate(zip(image_paths, is_normal_list)):
            try:
                # 加载原图（始终需要）
                image = Image.open(img_path).convert("RGB")

                mask_image = None
                if not is_normal:
                    if mask_paths is not None and mask_paths[i] and os.path.exists(mask_paths[i]):
                        mask_image = Image.open(mask_paths[i]).convert("RGB")
                    else:
                        print(f"异常样本缺少掩码图，跳过: {img_path}")
                        continue

                prompt = self.build_prompt(is_normal, has_mask=(mask_image is not None))

                # 构建多模态数据
                if is_normal:
                    multi_modal_data = {"image": [image]}
                else:
                    multi_modal_data = {"image": [image, mask_image]}

                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": multi_modal_data,
                    "original_index": i,
                    "image_path": img_path,
                    "is_normal": is_normal,
                    "original_image": image,        # 保留用于SAM3
                    "mask_path": mask_paths[i] if mask_paths else None
                })
                valid_indices.append(i)

            except Exception as e:
                print(f"加载图片 {img_path} 时出错: {e}")
                continue

        if not batch_inputs:
            return []

        sampling_params = self.get_sampling_params(temperature=temperature)
        outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)

        all_results = []

        # 第二遍：处理输出并可选地进行SAM3验证
        for result_idx, output in enumerate(outputs):
            if result_idx >= len(batch_inputs):
                break
            inp = batch_inputs[result_idx]
            orig_idx = inp["original_index"]
            img_path = inp["image_path"]
            is_normal = inp["is_normal"]
            original_image = inp["original_image"]
            mask_path = inp["mask_path"]

            if output and output.outputs:
                response = output.outputs[0].text.strip()
                parsed = self.parse_structured_output(response, is_normal)

                # 对于异常样本，进行SAM3自检
                sam3_verified = False
                sam3_bbox = None
                if not is_normal and self.use_sam3 and parsed["parsing_success"]:
                    # 使用description作为文本提示
                    description = parsed["description"]
                    if description:
                        verified, bbox, _, iou_val = self.verify_with_sam3(original_image, description, mask_path)
                        # 记录IoU
                        if iou_records is not None and iou_val >= 0:
                            iou_records.append({
                                "img_path": img_path,
                                "iou": iou_val,
                                "verified": verified,
                                "description": description,
                            })
                        if verified:
                            sam3_verified = True
                            sam3_bbox = bbox

                # 决定是否保留该样本
                if parsed["parsing_success"]:
                    if is_normal or (not is_normal and sam3_verified):
                        # 正常样本始终保留，异常样本需通过自检
                        result_entry = self.format_result_entry(
                            img_path, is_normal,
                            parsed["description"],
                            parsed["think"],
                            parsed["answer"],
                            temperature,
                            bbox=sam3_bbox if not is_normal else None
                        )
                        all_results.append(result_entry)
                    else:
                        # 自检未通过，丢弃
                        print(f"丢弃样本（自检失败）: {img_path}")
                else:
                    # 解析失败，记录但不保留（或可选择保留？根据需求，我们丢弃）
                    print(f"解析失败，丢弃: {img_path} (错误: {parsed['parsing_error']})")
            else:
                print(f"推理失败，丢弃: {img_path}")

        # 注意：这里我们没有处理被跳过的无效图片，因为已经在上面的循环中跳过了
        return all_results

    def format_result_entry(self, image_path: str, is_normal: bool,
                            description: str, think: str, answer: str, temperature: float,
                            bbox: Optional[List[float]] = None) -> Dict:
        """格式化结果条目，可选包含边界框"""
        desc_part = f"<description>{description}</description>" if description else ""
        assistant_content = f"{desc_part}<think>{think}</think><answer>{answer}</answer>"

        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nIs there any abnormality in the object in this picture? If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "images": [image_path]
        }

        # 添加元数据（调试用）
        entry["metadata"] = {
            "is_normal": is_normal,
            "parsing_status": "success",
            "parsing_success": True,
            "generation_temperature": temperature
        }
        # if bbox is not None:
        #     # entry["metadata"]["predicted_bbox"] = bbox
        #     # 也可以将bbox直接添加到assistant的content中，例如：
        #     entry["messages"][1]["content"] += f" <bbox>{bbox}</bbox>"

        return entry

    def remove_metadata_from_results(self, results: List[Dict]) -> List[Dict]:
        """移除metadata字段"""
        cleaned = []
        for res in results:
            cleaned.append({
                "messages": res["messages"],
                "images": res["images"]
            })
        return cleaned

    def filter_successful_results(self, results: List[Dict]) -> List[Dict]:
        """过滤出解析成功且通过自检的结果（实际上结果列表中已只包含成功）"""
        return [r for r in results if r.get("metadata", {}).get("parsing_success", False)]

    def _load_checkpoint(self, output_file):
        """加载断点续传的进度，返回 (已完成类别集合, 已有结果列表, 已有IoU记录列表, 统计信息)"""
        checkpoint_file = output_file + ".checkpoint.json"
        if not os.path.exists(checkpoint_file):
            return set(), [], [], {"total_processed": 0, "normal_kept": 0, "abnormal_kept": 0}

        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)

        done_categories = set(ckpt.get("done_categories", []))
        results = ckpt.get("results", [])
        iou_records = ckpt.get("iou_records", [])
        statistics = ckpt.get("statistics", {"total_processed": 0, "normal_kept": 0, "abnormal_kept": 0})

        print(f"从断点恢复：已完成 {len(done_categories)} 个类别，"
              f"已有 {len(results)} 条结果，{len(iou_records)} 条IoU记录")
        return done_categories, results, iou_records, statistics

    def _save_checkpoint(self, output_file, done_categories, results, iou_records, statistics):
        """保存当前进度到断点文件"""
        checkpoint_file = output_file + ".checkpoint.json"
        ckpt = {
            "done_categories": list(done_categories),
            "results": results,
            "iou_records": iou_records,
            "statistics": statistics,
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)

    def process_real_iad_dataset(self, dataset_path, meta_file_path,
                                output_file="outputs.json",
                                seed=42,
                                abnormal_only=False):
        """处理Real-IAD数据集，支持断点续传：每完成一个类别自动保存进度，崩溃后重跑自动跳过已完成的类别。"""
        print(f"开始处理数据集: {dataset_path}")
        print(f"仅异常样本: {'是' if abnormal_only else '否'}")

        random.seed(seed)
        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)

        # 加载断点
        done_categories, all_results, all_iou_records, statistics = self._load_checkpoint(output_file)

        start_time = time.time()
        categories = list(meta_data["test"].keys())
        print(f"类别: {categories}")

        for category in categories:
            if category in done_categories:
                print(f"\n跳过已完成类别: {category}")
                continue

            print(f"\n处理类别: {category}")
            test_data = meta_data["test"].get(category, [])
            if not test_data:
                done_categories.add(category)
                continue

            # 分离正常和异常样本
            normal_samples = []   # 路径
            abnormal_samples = [] # (img_path, mask_path)

            for item in test_data:
                img_path = os.path.join(dataset_path, item["img_path"])
                anomaly = item.get("anomaly", 0)
                if anomaly == 0:
                    normal_samples.append(img_path)
                else:
                    mask_path = item.get("mask_path", "")
                    if mask_path:
                        full_mask = os.path.join(dataset_path, mask_path)
                        if os.path.exists(full_mask):
                            abnormal_samples.append((img_path, full_mask))
                        else:
                            print(f"掩码不存在，跳过异常样本: {img_path}")

            print(f"正常: {len(normal_samples)}, 异常: {len(abnormal_samples)}")

            # 处理正常样本（无SAM3验证）
            if normal_samples and not abnormal_only:
                for start in tqdm(range(0, len(normal_samples), self.batch_size), desc=f"normal"):
                    batch = normal_samples[start:start+self.batch_size]
                    is_normal = [True]*len(batch)
                    results = self.batch_generate_structured_output(
                        batch, is_normal, mask_paths=None, temperature=0.0
                    )
                    all_results.extend(results)
                    statistics["total_processed"] += len(results)
                    statistics["normal_kept"] += len(results)

            # 处理异常样本（带SAM3验证）
            if abnormal_samples:
                img_list = [item[0] for item in abnormal_samples]
                mask_list = [item[1] for item in abnormal_samples]
                category_iou_records = []

                for start in tqdm(range(0, len(img_list), self.batch_size), desc=f"abnormal"):
                    batch_img = img_list[start:start+self.batch_size]
                    batch_mask = mask_list[start:start+self.batch_size]
                    is_normal = [False]*len(batch_img)
                    results = self.batch_generate_structured_output(
                        batch_img, is_normal, mask_paths=batch_mask, temperature=0.0,
                        iou_records=category_iou_records
                    )
                    all_results.extend(results)
                    statistics["total_processed"] += len(results)
                    statistics["abnormal_kept"] += len(results)

                # 为每条IoU记录添加类别信息
                for rec in category_iou_records:
                    rec["category"] = category
                all_iou_records.extend(category_iou_records)

            # 每完成一个类别，保存断点
            done_categories.add(category)
            self._save_checkpoint(output_file, done_categories, all_results, all_iou_records, statistics)
            print(f"类别 {category} 完成，进度已保存 ({len(done_categories)}/{len(categories)})")

        # 全部完成：保存最终结果
        cleaned = self.remove_metadata_from_results(all_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

        # 保存统计信息
        stats_file = "sampling_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)

        # 保存IoU记录并绘制可视化图
        if all_iou_records:
            output_dir = os.path.dirname(output_file) or "."
            iou_json_path = os.path.join(output_dir, "iou_records.json")
            with open(iou_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_iou_records, f, ensure_ascii=False, indent=2)
            print(f"IoU记录已保存至 {iou_json_path} (共 {len(all_iou_records)} 条)")

            plot_path = os.path.join(output_dir, "iou_statistics.png")
            self.plot_iou_statistics(all_iou_records, plot_path)

        # 删除断点文件
        checkpoint_file = output_file + ".checkpoint.json"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("断点文件已清理")

        print(f"\n处理完成！耗时 {time.time()-start_time:.2f} 秒")
        print(f"总保留样本: {statistics['total_processed']}")
        print(f"正常: {statistics['normal_kept']}, 异常: {statistics['abnormal_kept']}")
        print(f"结果保存至 {output_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="生成结构化输出并可选进行SAM3自检")
    parser.add_argument("--model-path", type=str, default="model/Qwen3-VL-32B")
    parser.add_argument("--dataset-path", type=str, default="data/Real-IAD")
    parser.add_argument("--meta-file", type=str, default="data/Real-IAD/meta.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--output-file", type=str, default="outputs.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-sam3", action="store_false", help="启用SAM3自检")
    parser.add_argument("--sam3-iou-threshold", type=float, default=0.0, help="SAM3验证的IoU阈值")
    parser.add_argument("--abnormal-only", action="store_true", help="仅生成异常样本描述，跳过正常样本")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"数据集路径不存在: {args.dataset_path}")
        return
    if not os.path.exists(args.meta_file):
        print(f"meta.json不存在: {args.meta_file}")
        return

    generator = StructuredOutputGenerator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        use_sam3=args.use_sam3,
        sam3_iou_threshold=args.sam3_iou_threshold
    )

    generator.process_real_iad_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file,
        seed=args.seed,
        abnormal_only=args.abnormal_only
    )


if __name__ == "__main__":
    main()
