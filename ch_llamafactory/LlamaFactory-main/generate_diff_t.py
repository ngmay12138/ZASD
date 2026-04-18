import os
import json
import torch
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import logging
import random
from typing import List, Dict

# vLLM相关导入
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class StructuredOutputGenerator:
    def __init__(self, model_path, device=None, batch_size=4, tensor_parallel_size=1):
        """
        初始化结构化输出生成器
        
        Args:
            model_path: Qwen3-VL模型路径
            device: 使用的设备，默认为cuda
            batch_size: 批处理大小
            tensor_parallel_size: GPU并行数量
        """
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"使用设备: {self.device}")
        print(f"批处理大小: {batch_size}")
        
        # 定义正常和异常图像的提示词
        self.normal_prompt = """
## Task Description
You are an industrial anomaly detection expert and need to explain why there is no abnormal situation in the objects in the image.
## Output Format Requirements
Answer strictly in the following format. Replace the description and think with your response, and do not add any additional text:
<description>Briefly describe the main features you observed in a few words</description><think>Based on the description,briefly describe your thought process about why the object in the image is normal</think><answer>normal</answer>
Now analyze this image:"""
        
        # 异常图像提示词：结合原图和掩码图
        self.abnormal_prompt_with_mask = """
## Task Description
You are an industrial anomaly detection expert. You are given two images: the first is the original object image, and the second is a defect mask image (highlighting the anomalous regions). Identify the defects, damage, abnormal states, or non-compliance issues in the object based on both images.
## Output Format Requirements
Answer strictly in the following format. Replace the description,location and think with your response, and do not add any additional text:
<description>Briefly describe in a few words the main features you observe in the original object image, focusing on the abnormal areas indicated by the mask.</description><location>Based on the mask, indicate the specific location of the abnormal areas in the original image</location><think>Briefly describe your thought process about why the object in the image is abnormal</think><answer>abnormal</answer>
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
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},  # 支持多张图片
        )
        
        print("vLLM引擎初始化完成！")
    
    def get_sampling_params(self, temperature=0.0):
        """获取不同temperature的采样参数"""
        return SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=1000,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
    
    def build_prompt(self, is_normal: bool, has_mask: bool = False) -> str:
        """
        根据图像类型构建提示词（文本部分）
        
        Args:
            is_normal: 是否为正常图像
            has_mask: 是否有掩码图（仅异常图像使用）
        
        Returns:
            提示词文本
        """
        if is_normal:
            prompt = self.normal_prompt
        else:
            # 异常图像使用带掩码的提示词
            prompt = self.abnormal_prompt_with_mask
        
        # 构建消息列表（图片占位符将由调用者添加）
        if is_normal:
            # 正常图像：单图片
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
            # 异常图像：双图片（原图+掩码图）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},      # 原图
                        {"type": "image"},      # 掩码图
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
        """解析结构化输出，只提取特定格式标签内的内容，不尝试修正任何错误"""
        result = {
            "raw_response": response,
            "description": "",
            "think": "",
            "answer": "",
            "parsing_success": False,
            "parsing_error": None
        }
     
        try:
            # 查找<description>标签的内容
            desc_start = response.find('<description>')
            desc_end = response.find('</description>')
            if desc_start != -1 and desc_end != -1 and desc_end > desc_start:
                desc_start += len('<description>')
                result["description"] = response[desc_start:desc_end].strip()
            
            # 查找<think>标签的内容
            think_start = response.find('<think>')
            think_end = response.find('</think>')
            if think_start != -1 and think_end != -1 and think_end > think_start:
                think_start += len('<think>')
                result["think"] = response[think_start:think_end].strip()
            
            # 查找<answer>标签的内容
            answer_start = response.find('<answer>')
            answer_end = response.find('</answer>')
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_start += len('<answer>')
                result["answer"] = response[answer_start:answer_end].strip()
            
            # 检查所有必需字段是否都成功解析
            if result["description"] and result["think"] and result["answer"]:
                # 验证answer是否为有效值
                if result["answer"].lower() in ["normal", "abnormal"]:
                    result["parsing_success"] = True
                else:
                    result["parsing_error"] = f"无效的answer值: {result['answer']}"
            else:
                # 记录具体哪个字段缺失
                missing_fields = []
                if not result["description"]:
                    missing_fields.append("description")
                if not result["think"]:
                    missing_fields.append("think")
                if not result["answer"]:
                    missing_fields.append("answer")
                result["parsing_error"] = f"缺失字段: {', '.join(missing_fields)}"
                    
        except Exception as e:
            result["parsing_error"] = f"解析异常: {str(e)}"
        
        return result
    
    def batch_generate_structured_output(self, image_paths: List[str], is_normal_list: List[bool], 
                                         mask_paths: List[str] = None, temperature=0.0) -> List[Dict]:
        """
        批量生成结构化输出
        
        Args:
            image_paths: 图片路径列表
            is_normal_list: 对应图片是否为正常的布尔值列表
            mask_paths: 掩码图路径列表（仅异常图片使用），长度需与image_paths一致，正常图片对应项可为None
            temperature: 生成温度
            
        Returns:
            results: 结构化输出结果列表（包含metadata）
        """
        if not image_paths:
            return []
        
        # 检查长度是否匹配
        if len(image_paths) != len(is_normal_list):
            raise ValueError(f"image_paths和is_normal_list长度不匹配: {len(image_paths)} != {len(is_normal_list)}")
        
        # 如果提供了mask_paths，确保长度一致
        if mask_paths is not None and len(mask_paths) != len(image_paths):
            raise ValueError(f"image_paths和mask_paths长度不匹配: {len(image_paths)} != {len(mask_paths)}")
        
        # 构建批次输入
        batch_inputs = []
        valid_indices = []
        
        for i, (image_path, is_normal) in enumerate(zip(image_paths, is_normal_list)):
            try:
                # 加载原图
                image = Image.open(image_path).convert("RGB")
                
                # 对于异常图片，需要加载掩码图
                mask_image = None
                if not is_normal:
                    # 获取掩码图路径
                    if mask_paths is not None and mask_paths[i] is not None:
                        mask_path = mask_paths[i]
                        if mask_path and os.path.exists(mask_path):
                            # 掩码图可能是单通道灰度图，转换为RGB以便模型处理
                            mask_img = Image.open(mask_path).convert("RGB")
                            mask_image = mask_img
                        else:
                            # 掩码图不存在，跳过此样本
                            print(f"警告: 掩码图不存在，跳过异常样本: {image_path}")
                            continue
                    else:
                        # 没有提供掩码图路径，跳过
                        print(f"警告: 异常样本缺少掩码图路径，跳过: {image_path}")
                        continue
                
                # 构建提示词（根据是否有掩码图决定图片数量）
                # 注意：build_prompt内部已经根据is_normal和has_mask构建正确的消息结构
                prompt = self.build_prompt(is_normal, has_mask=(mask_image is not None))
                
                # 构建多模态数据
                if is_normal:
                    # 正常图片：单张图片
                    multi_modal_data = {"image": [image]}
                else:
                    # 异常图片：两张图片（原图 + 掩码图）
                    multi_modal_data = {"image": [image, mask_image]}
                
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": multi_modal_data
                })
                valid_indices.append(i)
                
            except Exception as e:
                print(f"加载图片 {image_path} 时出错: {e}")
                # 跳过无效图片
        
        if not batch_inputs:
            return []
        
        # 使用指定temperature的采样参数
        sampling_params = self.get_sampling_params(temperature=temperature)
        
        # 批量推理
        outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
        
        # 解析结果
        all_results = []
        
        for result_idx, output in enumerate(outputs):
            if result_idx < len(valid_indices):
                orig_idx = valid_indices[result_idx]
                if output and output.outputs:
                    response = output.outputs[0].text.strip()
                    is_normal = is_normal_list[orig_idx]
                    
                    # 解析结构化输出
                    parsed_result = self.parse_structured_output(response, is_normal)
                    
                    # 构建结果条目
                    if parsed_result["parsing_success"]:
                        # 解析成功，构建正常结果
                        result_entry = self.format_result_entry(
                            image_paths[orig_idx], 
                            is_normal, 
                            parsed_result["description"],
                            parsed_result["think"],
                            parsed_result["answer"],
                            temperature
                        )
                    else:
                        # 解析失败，如实记录失败信息
                        result_entry = self.format_parsing_failed_entry(
                            image_paths[orig_idx], 
                            is_normal,
                            parsed_result["raw_response"],
                            parsed_result["parsing_error"],
                            temperature
                        )
                    
                    all_results.append(result_entry)
                else:
                    # 处理推理失败的情况
                    result_entry = self.format_error_entry(
                        image_paths[orig_idx], 
                        is_normal_list[orig_idx],
                        "inference_error",
                        temperature
                    )
                    all_results.append(result_entry)
        
        # 处理被跳过的无效图片（不在valid_indices中的）
        for i in range(len(image_paths)):
            if i not in valid_indices:
                # 对于无效图片，添加错误记录
                result_entry = self.format_error_entry(
                    image_paths[i], 
                    is_normal_list[i],
                    "image_loading_error",
                    temperature
                )
                all_results.append(result_entry)
        
        return all_results
    
    def format_result_entry(self, image_path: str, is_normal: bool, 
                           description: str, think: str, answer: str, temperature: float) -> Dict:
        """
        格式化结果条目，使其与mvtec_ad_example.json格式一致
        
        Args:
            image_path: 图片路径
            is_normal: 是否正常
            description: 描述文本
            think: 分析文本
            answer: 答案（normal/abnormal）
            temperature: 生成温度
            
        Returns:
            格式化后的结果条目（包含metadata）
        """
        # 构建完整的助手响应内容
        assistant_content = f"<description>{description}</description><think>{think}</think><answer>{answer}</answer>"
        
        # 构建与示例一致的格式
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
            "images": [image_path]  # 保持原始图像路径
        }
        
        # 添加元数据（只在内部处理时使用，不保存到最终输出）
        entry["metadata"] = {
            "is_normal": is_normal,
            "parsing_status": "success",
            "parsing_success": True,
            "generation_temperature": temperature
        }
        
        return entry
    
    def format_parsing_failed_entry(self, image_path: str, is_normal: bool, 
                                   raw_response: str, parsing_error: str, temperature: float) -> Dict:
        """
        格式化解析失败的条目，保持原始响应不变
        
        Args:
            image_path: 图片路径
            is_normal: 是否正常
            raw_response: 原始响应文本
            parsing_error: 解析错误信息
            temperature: 生成温度
            
        Returns:
            格式化后的解析失败条目（包含metadata）
        """
        # 保持原始响应不变
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nIs there any abnormality in the object in this picture? If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."
                },
                {
                    "role": "assistant",
                    "content": raw_response  # 保持原始响应，不尝试修正
                }
            ],
            "images": [image_path]
        }
        
        # 添加元数据（只在内部处理时使用，不保存到最终输出）
        entry["metadata"] = {
            "is_normal": is_normal,
            "parsing_status": "failed",
            "parsing_success": False,
            "parsing_error": parsing_error,
            "raw_response": raw_response,
            "generation_temperature": temperature
        }
        
        return entry
    
    def format_error_entry(self, image_path: str, is_normal: bool, error_type: str, temperature: float) -> Dict:
        """
        格式化错误条目
        
        Args:
            image_path: 图片路径
            is_normal: 是否正常
            error_type: 错误类型
            temperature: 生成温度
            
        Returns:
            格式化后的错误条目（包含metadata）
        """
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nIs there any abnormality in the object in this picture? If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."
                },
                {
                    "role": "assistant",
                    "content": f"<description>Error processing image</description><think>Failed to process image due to {error_type}</think><answer>error</answer>"
                }
            ],
            "images": [image_path]
        }
        
        # 添加错误元数据（只在内部处理时使用，不保存到最终输出）
        entry["metadata"] = {
            "is_normal": is_normal,
            "parsing_status": "error",
            "parsing_success": False,
            "error_type": error_type,
            "generation_temperature": temperature
        }
        
        return entry
    
    def remove_metadata_from_results(self, results: List[Dict]) -> List[Dict]:
        """
        从结果列表中移除metadata字段，仅保留messages和images
        
        Args:
            results: 包含metadata的结果列表
            
        Returns:
            不包含metadata的结果列表
        """
        cleaned_results = []
        for result in results:
            # 创建结果的副本，但不包含metadata字段
            cleaned_result = {
                "messages": result["messages"],
                "images": result["images"]
            }
            cleaned_results.append(cleaned_result)
        return cleaned_results
    
    def filter_successful_results(self, results: List[Dict]) -> List[Dict]:
        """
        过滤出解析成功的结果
        
        Args:
            results: 包含metadata的结果列表
            
        Returns:
            解析成功的结果列表
        """
        successful_results = []
        for result in results:
            # 检查结果是否解析成功
            if result.get("metadata", {}).get("parsing_success", False):
                successful_results.append(result)
        return successful_results
    
    def process_real_iad_dataset_with_sampling(self, dataset_path, meta_file_path, 
                                           low_temp_file="low_temperature_outputs.json",
                                           high_temp_file="high_temperature_outputs.json",
                                           split_ratio=0.5, 
                                           low_temperature=0.0,
                                           high_temperature=1.0,
                                           seed=42):
        """
        处理整个Real-IAD数据集，随机采样并分别使用不同temperature生成结构化输出
        
        Args:
            dataset_path: Real-IAD数据集根目录
            meta_file_path: meta.json文件路径
            low_temp_file: 低温结果输出文件
            high_temp_file: 高温结果输出文件
            split_ratio: 低温组数据比例（剩余为高温组）
            low_temperature: 低温组的temperature值
            high_temperature: 高温组的temperature值
            seed: 随机种子
            
        Returns:
            low_temp_results, high_temp_results: 低温组和高温组的结果列表（包含metadata）
        """
        print(f"开始处理Real-IAD数据集: {dataset_path}")
        print(f"低温组比例: {split_ratio}, 低温: {low_temperature}, 高温: {high_temperature}")
        
        # 设置随机种子
        random.seed(seed)
        
        # 加载meta.json文件
        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)
        
        # 存储最终结果（包含metadata，用于内部处理和统计）
        low_temp_results_with_metadata = []
        high_temp_results_with_metadata = []
        
        # 统计信息
        statistics = {
            "total_images": 0,
            "low_temperature_images": 0,
            "high_temperature_images": 0,
            "normal_images": 0,
            "abnormal_images": 0,
            "low_temp_successful_parsing": 0,
            "low_temp_failed_parsing": 0,
            "high_temp_successful_parsing": 0,
            "high_temp_failed_parsing": 0,
            "parsing_failed_details": []
        }
        
        import time
        start_time = time.time()
        
        # 获取所有类别
        categories = list(meta_data["test"].keys())
        print(f"找到的类别: {categories}")
        
        # 遍历每个类别
        for category in categories:
            print(f"\n处理类别: {category}")
            
            # 从meta.json中获取该类别下的所有测试数据
            test_data_list = meta_data["test"].get(category, [])
            
            if not test_data_list:
                print(f"警告: {category} 类别没有测试数据，跳过")
                continue
            
            # 分离正常和异常样本，并同时收集异常样本的掩码路径
            normal_samples = []  # 每个元素为 (img_path,)
            abnormal_samples = []  # 每个元素为 (img_path, mask_path)
            
            for item in test_data_list:
                img_path = item.get("img_path", "")
                mask_path = item.get("mask_path", "")
                anomaly = item.get("anomaly", 0)
                
                # 构建完整路径
                full_img_path = os.path.join(dataset_path, img_path)
                
                if anomaly == 0:
                    # 正常样本
                    normal_samples.append(full_img_path)
                else:
                    # 异常样本：检查掩码图是否存在
                    if mask_path:
                        full_mask_path = os.path.join(dataset_path, mask_path)
                        if os.path.exists(full_mask_path):
                            abnormal_samples.append((full_img_path, full_mask_path))
                        else:
                            print(f"警告: 掩码图不存在，跳过异常样本: {full_img_path}")
                    else:
                        print(f"警告: 异常样本缺少mask_path，跳过: {full_img_path}")
            
            print(f"  正常图像: {len(normal_samples)} 张, 异常图像: {len(abnormal_samples)} 张")
            
            # 处理正常图像
            if normal_samples:
                print(f"  处理正常图像...")
                
                # 随机打乱图像列表
                random.shuffle(normal_samples)
                
                # 按比例分割数据
                split_point = int(len(normal_samples) * split_ratio)
                low_temp_files = normal_samples[:split_point]
                high_temp_files = normal_samples[split_point:]
                
                print(f"    低温组: {len(low_temp_files)} 张, 高温组: {len(high_temp_files)} 张")
                
                # 处理低温组
                if low_temp_files:
                    print(f"    生成低温组结果 (temperature={low_temperature})...")
                    
                    # 按批次处理低温组图片
                    for batch_start in tqdm(range(0, len(low_temp_files), self.batch_size), 
                                           desc=f"  low_temp(good)"):
                        batch_files = low_temp_files[batch_start:batch_start + self.batch_size]
                        # 创建对应批次的is_normal_list
                        batch_is_normal = [True] * len(batch_files)
                        
                        # 批量生成结构化输出（正常图像不需要掩码）
                        batch_results = self.batch_generate_structured_output(
                            batch_files, batch_is_normal, mask_paths=None, temperature=low_temperature
                        )
                        
                        # 添加到低温结果列表
                        low_temp_results_with_metadata.extend(batch_results)
                        
                        # 更新统计
                        statistics["total_images"] += len(batch_results)
                        statistics["low_temperature_images"] += len(batch_results)
                        statistics["normal_images"] += len(batch_results)
                        
                        # 统计解析成功/失败
                        for result in batch_results:
                            if result.get("metadata", {}).get("parsing_success", False):
                                statistics["low_temp_successful_parsing"] += 1
                            else:
                                statistics["low_temp_failed_parsing"] += 1
                                # 记录解析失败详情
                                parsing_error = result.get("metadata", {}).get("parsing_error", "unknown")
                                statistics["parsing_failed_details"].append({
                                    "image": result["images"][0] if result.get("images") else "unknown",
                                    "error": parsing_error,
                                    "category": category,
                                    "type": "normal",
                                    "temperature_group": "low"
                                })
                
                # 处理高温组
                if high_temp_files:
                    print(f"    生成高温组结果 (temperature={high_temperature})...")
                    
                    # 按批次处理高温组图片
                    for batch_start in tqdm(range(0, len(high_temp_files), self.batch_size), 
                                           desc=f"  high_temp(good)"):
                        batch_files = high_temp_files[batch_start:batch_start + self.batch_size]
                        # 创建对应批次的is_normal_list
                        batch_is_normal = [True] * len(batch_files)
                        
                        # 批量生成结构化输出
                        batch_results = self.batch_generate_structured_output(
                            batch_files, batch_is_normal, mask_paths=None, temperature=high_temperature
                        )
                        
                        # 添加到高温结果列表
                        high_temp_results_with_metadata.extend(batch_results)
                        
                        # 更新统计
                        statistics["total_images"] += len(batch_results)
                        statistics["high_temperature_images"] += len(batch_results)
                        statistics["normal_images"] += len(batch_results)
                        
                        # 统计解析成功/失败
                        for result in batch_results:
                            if result.get("metadata", {}).get("parsing_success", False):
                                statistics["high_temp_successful_parsing"] += 1
                            else:
                                statistics["high_temp_failed_parsing"] += 1
                                # 记录解析失败详情
                                parsing_error = result.get("metadata", {}).get("parsing_error", "unknown")
                                statistics["parsing_failed_details"].append({
                                    "image": result["images"][0] if result.get("images") else "unknown",
                                    "error": parsing_error,
                                    "category": category,
                                    "type": "normal",
                                    "temperature_group": "high"
                                })
                
                print(f"    完成: {len(normal_samples)} 张正常图像")
            
            # 处理异常图像
            if abnormal_samples:
                print(f"  处理异常图像...")
                
                # 分离路径和掩码路径
                abnormal_img_paths = [item[0] for item in abnormal_samples]
                abnormal_mask_paths = [item[1] for item in abnormal_samples]
                
                # 随机打乱图像列表（保持对应关系）
                combined = list(zip(abnormal_img_paths, abnormal_mask_paths))
                random.shuffle(combined)
                abnormal_img_paths, abnormal_mask_paths = zip(*combined) if combined else ([], [])
                abnormal_img_paths = list(abnormal_img_paths)
                abnormal_mask_paths = list(abnormal_mask_paths)
                
                # 按比例分割数据
                split_point = int(len(abnormal_img_paths) * split_ratio)
                low_temp_img_files = abnormal_img_paths[:split_point]
                low_temp_mask_files = abnormal_mask_paths[:split_point]
                high_temp_img_files = abnormal_img_paths[split_point:]
                high_temp_mask_files = abnormal_mask_paths[split_point:]
                
                print(f"    低温组: {len(low_temp_img_files)} 张, 高温组: {len(high_temp_img_files)} 张")
                
                # 处理低温组
                if low_temp_img_files:
                    print(f"    生成低温组结果 (temperature={low_temperature})...")
                    
                    # 按批次处理低温组图片
                    for batch_start in tqdm(range(0, len(low_temp_img_files), self.batch_size), 
                                           desc=f"  low_temp(abnormal)"):
                        batch_img_files = low_temp_img_files[batch_start:batch_start + self.batch_size]
                        batch_mask_files = low_temp_mask_files[batch_start:batch_start + self.batch_size]
                        # 创建对应批次的is_normal_list
                        batch_is_normal = [False] * len(batch_img_files)
                        
                        # 批量生成结构化输出（异常图像需要提供掩码路径）
                        batch_results = self.batch_generate_structured_output(
                            batch_img_files, batch_is_normal, mask_paths=batch_mask_files, temperature=low_temperature
                        )
                        
                        # 添加到低温结果列表
                        low_temp_results_with_metadata.extend(batch_results)
                        
                        # 更新统计
                        statistics["total_images"] += len(batch_results)
                        statistics["low_temperature_images"] += len(batch_results)
                        statistics["abnormal_images"] += len(batch_results)
                        
                        # 统计解析成功/失败
                        for result in batch_results:
                            if result.get("metadata", {}).get("parsing_success", False):
                                statistics["low_temp_successful_parsing"] += 1
                            else:
                                statistics["low_temp_failed_parsing"] += 1
                                # 记录解析失败详情
                                parsing_error = result.get("metadata", {}).get("parsing_error", "unknown")
                                statistics["parsing_failed_details"].append({
                                    "image": result["images"][0] if result.get("images") else "unknown",
                                    "error": parsing_error,
                                    "category": category,
                                    "type": "abnormal",
                                    "temperature_group": "low"
                                })
                
                # 处理高温组
                if high_temp_img_files:
                    print(f"    生成高温组结果 (temperature={high_temperature})...")
                    
                    # 按批次处理高温组图片
                    for batch_start in tqdm(range(0, len(high_temp_img_files), self.batch_size), 
                                           desc=f"  high_temp(abnormal)"):
                        batch_img_files = high_temp_img_files[batch_start:batch_start + self.batch_size]
                        batch_mask_files = high_temp_mask_files[batch_start:batch_start + self.batch_size]
                        # 创建对应批次的is_normal_list
                        batch_is_normal = [False] * len(batch_img_files)
                        
                        # 批量生成结构化输出
                        batch_results = self.batch_generate_structured_output(
                            batch_img_files, batch_is_normal, mask_paths=batch_mask_files, temperature=high_temperature
                        )
                        
                        # 添加到高温结果列表
                        high_temp_results_with_metadata.extend(batch_results)
                        
                        # 更新统计
                        statistics["total_images"] += len(batch_results)
                        statistics["high_temperature_images"] += len(batch_results)
                        statistics["abnormal_images"] += len(batch_results)
                        
                        # 统计解析成功/失败
                        for result in batch_results:
                            if result.get("metadata", {}).get("parsing_success", False):
                                statistics["high_temp_successful_parsing"] += 1
                            else:
                                statistics["high_temp_failed_parsing"] += 1
                                # 记录解析失败详情
                                parsing_error = result.get("metadata", {}).get("parsing_error", "unknown")
                                statistics["parsing_failed_details"].append({
                                    "image": result["images"][0] if result.get("images") else "unknown",
                                    "error": parsing_error,
                                    "category": category,
                                    "type": "abnormal",
                                    "temperature_group": "high"
                                })
                
                print(f"    完成: {len(abnormal_img_paths)} 张异常图像")
        
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算成功率
        if statistics["total_images"] > 0:
            low_temp_success_rate = statistics["low_temp_successful_parsing"] / statistics["low_temperature_images"] if statistics["low_temperature_images"] > 0 else 0
            high_temp_success_rate = statistics["high_temp_successful_parsing"] / statistics["high_temperature_images"] if statistics["high_temperature_images"] > 0 else 0
            overall_success_rate = (statistics["low_temp_successful_parsing"] + statistics["high_temp_successful_parsing"]) / statistics["total_images"]
            
            print(f"\n{'='*50}")
            print(f"处理完成!")
            print(f"总图片数: {statistics['total_images']}")
            print(f"低温组图片: {statistics['low_temperature_images']}")
            print(f"高温组图片: {statistics['high_temperature_images']}")
            print(f"正常图片: {statistics['normal_images']}")
            print(f"异常图片: {statistics['abnormal_images']}")
            print(f"低温组成功解析: {statistics['low_temp_successful_parsing']}")
            print(f"低温组解析失败: {statistics['low_temp_failed_parsing']}")
            print(f"低温组解析成功率: {low_temp_success_rate:.4f}")
            print(f"高温组成功解析: {statistics['high_temp_successful_parsing']}")
            print(f"高温组解析失败: {statistics['high_temp_failed_parsing']}")
            print(f"高温组解析成功率: {high_temp_success_rate:.4f}")
            print(f"总体解析成功率: {overall_success_rate:.4f}")
            print(f"处理时间: {processing_time:.2f} 秒")
            print(f"{'='*50}")
            
            # 输出解析失败详情
            if statistics["parsing_failed_details"]:
                print(f"\n解析失败详情 (前10个):")
                for i, detail in enumerate(statistics["parsing_failed_details"][:10]):
                    print(f"  {i+1}. 图片: {os.path.basename(detail['image'])}")
                    print(f"     错误: {detail['error']}")
                    print(f"     类别: {detail['category']}, 类型: {detail['type']}, 温度组: {detail['temperature_group']}")
                if len(statistics["parsing_failed_details"]) > 10:
                    print(f"  ... 还有 {len(statistics['parsing_failed_details']) - 10} 个解析失败记录")
        
        # 从结果中过滤出解析成功的结果
        successful_low_temp_results = self.filter_successful_results(low_temp_results_with_metadata)
        successful_high_temp_results = self.filter_successful_results(high_temp_results_with_metadata)
        
        # 从结果中移除metadata字段，只保留messages和images
        successful_low_temp_results_without_metadata = self.remove_metadata_from_results(successful_low_temp_results)
        successful_high_temp_results_without_metadata = self.remove_metadata_from_results(successful_high_temp_results)
        
        # 保存低温结果到文件（不包含metadata，只保存解析成功的结果）
        with open(low_temp_file, 'w', encoding='utf-8') as f:
            json.dump(successful_low_temp_results_without_metadata, f, ensure_ascii=False, indent=2)
        print(f"\n低温组结构化输出已保存到: {low_temp_file} (只包含{len(successful_low_temp_results_without_metadata)}个解析成功的条目)")
        
        # 保存高温结果到文件（不包含metadata，只保存解析成功的结果）
        with open(high_temp_file, 'w', encoding='utf-8') as f:
            json.dump(successful_high_temp_results_without_metadata, f, ensure_ascii=False, indent=2)
        print(f"高温组结构化输出已保存到: {high_temp_file} (只包含{len(successful_high_temp_results_without_metadata)}个解析成功的条目)")
        
        # 保存统计信息到单独的文件（包含metadata信息）
        stats_file = "sampling_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存到: {stats_file}")
        
        # 也保存包含metadata的版本以供调试（包含所有结果，包括解析失败的）
        metadata_low_temp_file = low_temp_file.replace('.json', '_with_metadata.json')
        metadata_high_temp_file = high_temp_file.replace('.json', '_with_metadata.json')
        
        with open(metadata_low_temp_file, 'w', encoding='utf-8') as f:
            json.dump(low_temp_results_with_metadata, f, ensure_ascii=False, indent=2)
        
        with open(metadata_high_temp_file, 'w', encoding='utf-8') as f:
            json.dump(high_temp_results_with_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"包含metadata的低温组结果已保存到: {metadata_low_temp_file} (包含所有{len(low_temp_results_with_metadata)}个条目)")
        print(f"包含metadata的高温组结果已保存到: {metadata_high_temp_file} (包含所有{len(high_temp_results_with_metadata)}个条目)")
        
        return low_temp_results_with_metadata, high_temp_results_with_metadata


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="为Real-IAD数据集生成结构化输出（带随机采样）")
    parser.add_argument("--model-path", type=str, default="model/Qwen3-VL-32B")
    parser.add_argument("--dataset-path", type=str, default="data/Real-IAD")
    parser.add_argument("--meta-file", type=str, default="data/Real-IAD/meta.json")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批处理大小")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="GPU并行数量")
    parser.add_argument("--low-temp-file", type=str, default="t0_with_mask.json",
                        help="低温组输出结果文件")
    parser.add_argument("--high-temp-file", type=str, default="high_temperature_outputs.json",
                        help="高温组输出结果文件")
    parser.add_argument("--split-ratio", type=float, default=1,
                        help="低温组数据比例（0-1之间）")
    parser.add_argument("--low-temperature", type=float, default=0.0,
                        help="低温组的temperature值")
    parser.add_argument("--high-temperature", type=float, default=1.0,
                        help="高温组的temperature值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return
    
    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json文件 {args.meta_file} 不存在！")
        return
    
    # 检查split_ratio是否在有效范围内
    if args.split_ratio < 0 or args.split_ratio > 1:
        print(f"错误: split_ratio必须在0到1之间！")
        return
    
    # 创建生成器实例
    print(f"使用vLLM加载模型: {args.model_path}")
    generator = StructuredOutputGenerator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # 处理Real-IAD数据集，带随机采样和不同temperature生成
    low_temp_results, high_temp_results = generator.process_real_iad_dataset_with_sampling(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        low_temp_file=args.low_temp_file,
        high_temp_file=args.high_temp_file,
        split_ratio=args.split_ratio,
        low_temperature=args.low_temperature,
        high_temperature=args.high_temperature,
        seed=args.seed
    )


if __name__ == "__main__":
    main()