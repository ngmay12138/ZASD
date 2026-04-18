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
from typing import List, Dict

# vLLM相关导入
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class MVTecStructuredOutputGenerator:
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
        self.normal_prompt = """Please carefully analyze the objects in this image and explain why there is no abnormality.
## Task Description
You are an anomaly detection expert and need to explain why there is no abnormal situation in the objects in the image.
## Output Format Requirements
Answer strictly in the following format. Replace the description and analysis with your response, and do not add any additional text:
<description>Briefly describe the key features you observe</description><analysis>Analysis of why the observed object is normal</analysis><answer>normal</answer>
Now analyze this image:"""
        
        self.abnormal_prompt = """Please carefully analyze the objects in this image and explain why there is an abnormality.
## Task Description
You are an anomaly detection expert and need to identify defects, damage, abnormal states, or non-compliance issues in objects in images.
## Output Format Requirements
Answer strictly in the following format. Replace the description, analysis, and location with your response, and do not add any additional text:
<description>Briefly describe the key features you observe</description><analysis>Analysis of why the observed object is an anomaly</analysis><location>The specific location of the anomaly</location><answer>abnormal</answer>
Now analyze this image:"""
        
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
            limit_mm_per_prompt={"image": 10},
        )
        
        # 设置生成参数
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=1000,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        print("vLLM引擎初始化完成！")
    
    def build_prompt(self, is_normal: bool) -> str:
        """根据图像类型构建提示词"""
        if is_normal:
            prompt = self.normal_prompt
        else:
            prompt = self.abnormal_prompt
            
        messages = [
            {
                "role": "user",
                "content": [
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
        """解析结构化输出，只提取特定格式标签内的内容"""
        result = {
            "raw_response": response,
            "description": "",
            "analysis": "",
            "location": "",
            "answer": "",
            "parsing_success": False
        }
     
        try:
            # 使用正则表达式或直接字符串查找来解析标签
            # 先尝试使用更精确的查找方法
            
            # 查找<description>标签的内容
            desc_start = response.find('<description>')
            desc_end = response.find('</description>')
            if desc_start != -1 and desc_end != -1 and desc_end > desc_start:
                desc_start += len('<description>')
                # 提取<description>和</description>之间的内容，包括可能的引号
                result["description"] = response[desc_start:desc_end].strip()
            
            # 查找<analysis>标签的内容
            analysis_start = response.find('<analysis>')
            analysis_end = response.find('</analysis>')
            if analysis_start != -1 and analysis_end != -1 and analysis_end > analysis_start:
                analysis_start += len('<analysis>')
                result["analysis"] = response[analysis_start:analysis_end].strip()
            
            # 查找<location>标签的内容（仅异常图像）
            if not is_normal:
                location_start = response.find('<location>')
                location_end = response.find('</location>')
                if location_start != -1 and location_end != -1 and location_end > location_start:
                    location_start += len('<location>')
                    result["location"] = response[location_start:location_end].strip()
            
            # 查找<answer>标签的内容
            answer_start = response.find('<answer>')
            answer_end = response.find('</answer>')
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_start += len('<answer>')
                result["answer"] = response[answer_start:answer_end].strip()
            
            # 检查所有必需字段是否都成功解析
            if result["description"] and result["analysis"] and result["answer"]:
                if is_normal or (not is_normal and result["location"]):
                    result["parsing_success"] = True
                    
        except Exception as e:
            print(f"解析输出时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def batch_generate_structured_output(self, image_paths: List[str], is_normal_list: List[bool]) -> List[Dict]:
        """
        批量生成结构化输出
        
        Args:
            image_paths: 图片路径列表
            is_normal_list: 对应图片是否为正常的布尔值列表
            
        Returns:
            results: 结构化输出结果列表
        """
        if not image_paths:
            return []
        
        # 检查长度是否匹配
        if len(image_paths) != len(is_normal_list):
            raise ValueError("image_paths和is_normal_list长度不匹配")
        
        # 构建批次输入
        batch_inputs = []
        valid_indices = []
        
        for i, (image_path, is_normal) in enumerate(zip(image_paths, is_normal_list)):
            try:
                # 加载图片
                image = Image.open(image_path).convert("RGB")
                
                # 构建提示词
                prompt = self.build_prompt(is_normal)
                
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": [image]}
                })
                valid_indices.append(i)
                
            except Exception as e:
                print(f"加载图片 {image_path} 时出错: {e}")
                # 跳过无效图片
        
        if not batch_inputs:
            return []
        
        # 批量推理
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        # 解析结果
        all_results = []
        result_idx = 0
        
        for i in range(len(image_paths)):
            if i in valid_indices:
                if result_idx < len(outputs) and outputs[result_idx] and outputs[result_idx].outputs:
                    response = outputs[result_idx].outputs[0].text.strip()
                    is_normal = is_normal_list[i]
                    
                    # 解析结构化输出
                    parsed_result = self.parse_structured_output(response, is_normal)
                    
                    # 调试信息：打印解析结果
                    if not parsed_result["parsing_success"]:
                        print(f"解析失败，原始响应: {response}")
                        print(f"解析结果: {parsed_result}")
                    
                    result = {
                        "image_path": image_paths[i],
                        "is_normal": is_normal,
                        "response": response,
                        "parsed": parsed_result
                    }
                    
                    all_results.append(result)
                    result_idx += 1
                else:
                    # 处理推理失败的情况
                    all_results.append({
                        "image_path": image_paths[i],
                        "is_normal": is_normal_list[i],
                        "response": "error",
                        "parsed": {
                            "raw_response": "error",
                            "description": "",
                            "analysis": "",
                            "location": "",
                            "answer": "",
                            "parsing_success": False
                        }
                    })
                    result_idx += 1
            else:
                # 对于无效图片，添加错误记录
                all_results.append({
                    "image_path": image_paths[i],
                    "is_normal": is_normal_list[i],
                    "response": "image loading error",
                    "parsed": {
                        "raw_response": "image loading error",
                        "description": "",
                        "analysis": "",
                        "location": "",
                        "answer": "",
                        "parsing_success": False
                    }
                })
        
        return all_results
    
    def process_mvtec_dataset(self, dataset_path, meta_file_path, output_file="structured_outputs.json"):
        """
        处理整个MVTec AD数据集，生成结构化输出
        
        Args:
            dataset_path: MVTec AD数据集根目录
            meta_file_path: meta.json文件路径
            output_file: 结果输出文件
            
        Returns:
            results: 处理结果字典
        """
        print(f"开始处理数据集: {dataset_path}")
        
        # 加载meta.json文件
        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)
        
        # 存储结果
        results = {
            "dataset_info": {
                "name": "MVTec AD",
                "path": dataset_path,
                "processing_time": None
            },
            "categories": {},
            "statistics": {
                "total_images": 0,
                "normal_images": 0,
                "abnormal_images": 0,
                "successful_parsing": 0,
                "failed_parsing": 0
            }
        }
        
        import time
        start_time = time.time()
        
        # 获取所有类别
        categories = list(meta_data["train"].keys())
        print(f"找到的类别: {categories}")
        
        # 遍历每个类别
        for category in categories:
            print(f"\n处理类别: {category}")
            
            # 初始化类别结果
            results["categories"][category] = {
                "normal": [],
                "abnormal": []
            }
            
            # 构建test目录路径
            test_base_path = os.path.join(dataset_path, category, "test")
            
            if not os.path.exists(test_base_path):
                print(f"警告: {test_base_path} 不存在，跳过此类别")
                continue
            
            # 获取所有子目录（包括good和不同的缺陷类型）
            subdirs = [d for d in os.listdir(test_base_path) 
                      if os.path.isdir(os.path.join(test_base_path, d))]
            
            # 先处理正常图像（good子目录）
            if "good" in subdirs:
                print(f"  处理正常图像 (good)...")
                good_path = os.path.join(test_base_path, "good")
                image_files = glob.glob(os.path.join(good_path, "*.png"))
                
                # 按批次处理正常图片
                for batch_start in tqdm(range(0, len(image_files), self.batch_size), 
                                       desc=f"  good"):
                    batch_files = image_files[batch_start:batch_start + self.batch_size]
                    # 所有正常图像
                    is_normal_list = [True] * len(batch_files)
                    
                    # 批量生成结构化输出
                    batch_results = self.batch_generate_structured_output(batch_files, is_normal_list)
                    
                    # 处理结果
                    for result in batch_results:
                        results["categories"][category]["normal"].append(result)
                        
                        # 更新统计
                        results["statistics"]["total_images"] += 1
                        results["statistics"]["normal_images"] += 1
                        if result["parsed"]["parsing_success"]:
                            results["statistics"]["successful_parsing"] += 1
                        else:
                            results["statistics"]["failed_parsing"] += 1
                
                print(f"    完成: {len(image_files)} 张正常图像")
            
            # 处理异常图像（非good子目录）
            abnormal_subdirs = [d for d in subdirs if d != "good"]
            
            for subdir in abnormal_subdirs:
                print(f"  处理异常图像 ({subdir})...")
                subdir_path = os.path.join(test_base_path, subdir)
                image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                
                # 按批次处理异常图片
                for batch_start in tqdm(range(0, len(image_files), self.batch_size), 
                                       desc=f"  {subdir}"):
                    batch_files = image_files[batch_start:batch_start + self.batch_size]
                    # 所有异常图像
                    is_normal_list = [False] * len(batch_files)
                    
                    # 批量生成结构化输出
                    batch_results = self.batch_generate_structured_output(batch_files, is_normal_list)
                    
                    # 处理结果
                    for result in batch_results:
                        result["defect_type"] = subdir  # 添加缺陷类型
                        results["categories"][category]["abnormal"].append(result)
                        
                        # 更新统计
                        results["statistics"]["total_images"] += 1
                        results["statistics"]["abnormal_images"] += 1
                        if result["parsed"]["parsing_success"]:
                            results["statistics"]["successful_parsing"] += 1
                        else:
                            results["statistics"]["failed_parsing"] += 1
                
                print(f"    完成: {len(image_files)} 张异常图像 ({subdir})")
        
        # 计算处理时间
        end_time = time.time()
        results["dataset_info"]["processing_time"] = end_time - start_time
        
        # 计算成功率
        if results["statistics"]["total_images"] > 0:
            success_rate = results["statistics"]["successful_parsing"] / results["statistics"]["total_images"]
            print(f"\n{'='*50}")
            print(f"处理完成!")
            print(f"总图片数: {results['statistics']['total_images']}")
            print(f"正常图片: {results['statistics']['normal_images']}")
            print(f"异常图片: {results['statistics']['abnormal_images']}")
            print(f"成功解析: {results['statistics']['successful_parsing']}")
            print(f"解析失败: {results['statistics']['failed_parsing']}")
            print(f"解析成功率: {success_rate:.4f}")
            print(f"处理时间: {results['dataset_info']['processing_time']:.2f} 秒")
            print(f"{'='*50}")
        
        # 保存结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结构化输出已保存到: {output_file}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="为MVTec AD数据集生成结构化输出")
    parser.add_argument("--model-path", type=str, default="model/Qwen3-VL-32B")
    parser.add_argument("--dataset-path", type=str, default="data/mvtec_ad")
    parser.add_argument("--meta-file", type=str, default="data/mvtec_ad/meta.json")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批处理大小")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="GPU并行数量")
    parser.add_argument("--output-file", type=str, default="32b_mvtec_0.6t.json",
                        help="输出结果文件")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return
    
    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json文件 {args.meta_file} 不存在！")
        return
    
    # 创建生成器实例
    print(f"使用vLLM加载模型: {args.model_path}")
    generator = MVTecStructuredOutputGenerator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # 处理数据集，生成结构化输出
    results = generator.process_mvtec_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
################################################################################################
##提取失败样本
# import json
# import argparse
# from typing import Dict, List, Any
# from pathlib import Path
# import os

# def extract_failed_samples(json_file_path: str, output_file: str = "failed_samples.json"):
#     """
#     从结构化输出JSON文件中提取所有解析失败的样本
    
#     Args:
#         json_file_path: 输入JSON文件路径
#         output_file: 输出JSON文件路径
#     """
    
#     print(f"正在读取文件: {json_file_path}")
    
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"错误: 文件 {json_file_path} 不存在")
#         return
#     except json.JSONDecodeError as e:
#         print(f"错误: JSON文件解析失败: {e}")
#         return
    
#     # 检查数据结构
#     if "categories" not in data:
#         print("错误: JSON文件中缺少'categories'字段")
#         return
    
#     # 收集所有解析失败的样本
#     failed_samples = []
    
#     # 统计信息
#     stats = {
#         "total_failed": 0,
#         "failed_by_category": {},
#         "failed_normal": 0,
#         "failed_abnormal": 0,
#         "failed_by_parsing_success": 0,
#         "failed_by_error_response": 0
#     }
    
#     # 遍历所有类别
#     for category, category_data in data["categories"].items():
#         category_failed = []
        
#         # 处理正常样本
#         for sample in category_data.get("normal", []):
#             if is_sample_failed(sample):
#                 sample_info = create_sample_info(category, sample, is_normal=True)
#                 category_failed.append(sample_info)
#                 failed_samples.append(sample_info)
                
#                 # 更新统计
#                 stats["total_failed"] += 1
#                 stats["failed_normal"] += 1
#                 update_error_stats(stats, sample)
        
#         # 处理异常样本
#         for sample in category_data.get("abnormal", []):
#             if is_sample_failed(sample):
#                 sample_info = create_sample_info(category, sample, is_normal=False)
#                 category_failed.append(sample_info)
#                 failed_samples.append(sample_info)
                
#                 # 更新统计
#                 stats["total_failed"] += 1
#                 stats["failed_abnormal"] += 1
#                 update_error_stats(stats, sample)
        
#         # 记录类别失败统计
#         if category_failed:
#             stats["failed_by_category"][category] = len(category_failed)
    
#     # 输出统计信息
#     print("\n" + "="*60)
#     print("解析失败样本统计")
#     print("="*60)
#     print(f"总失败样本数: {stats['total_failed']}")
#     print(f"正常样本失败数: {stats['failed_normal']}")
#     print(f"异常样本失败数: {stats['failed_abnormal']}")
#     print(f"因解析成功字段失败: {stats['failed_by_parsing_success']}")
#     print(f"因错误响应失败: {stats['failed_by_error_response']}")
    
#     if stats["failed_by_category"]:
#         print("\n按类别统计:")
#         for category, count in stats["failed_by_category"].items():
#             print(f"  {category}: {count}")
    
#     # 如果没有失败样本
#     if not failed_samples:
#         print("\n✓ 恭喜！所有样本都解析成功！")
#         return
    
#     # 创建输出数据结构
#     output_data = {
#         "source_file": json_file_path,
#         "extraction_time": None,  # 将在后面添加
#         "statistics": stats,
#         "failed_samples": failed_samples
#     }
    
#     # 添加提取时间
#     from datetime import datetime
#     output_data["extraction_time"] = datetime.now().isoformat()
    
#     # 保存到输出文件
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, ensure_ascii=False, indent=2)
#         print(f"\n✓ 解析失败的样本已保存到: {output_file}")
#         print(f"✓ 共提取了 {len(failed_samples)} 个失败样本")
        
#         # 可选：打印前5个失败样本的摘要
#         if failed_samples:
#             print("\n前5个失败样本:")
#             for i, sample in enumerate(failed_samples[:5]):
#                 print(f"\n{i+1}. 类别: {sample['category']}")
#                 print(f"   图像: {sample['image_path']}")
#                 print(f"   类型: {'正常' if sample['is_normal'] else '异常'}")
#                 print(f"   响应状态: {sample['response_status']}")
#                 if sample['error_reason']:
#                     print(f"   失败原因: {sample['error_reason']}")
                
#     except Exception as e:
#         print(f"错误: 保存输出文件失败: {e}")


# def is_sample_failed(sample: Dict[str, Any]) -> bool:
#     """
#     判断样本是否解析失败
    
#     失败条件：
#     1. parsed.parsing_success 为 False
#     2. response 包含错误信息（如 "error"）
#     3. 缺少必要的解析字段
#     """
    
#     # 检查响应字段
#     response = sample.get("response", "").lower()
#     if response in ["error", "image loading error", ""]:
#         return True
    
#     # 检查解析结果
#     parsed = sample.get("parsed", {})
    
#     # 如果parsing_success为False，则失败
#     if not parsed.get("parsing_success", False):
#         return True
    
#     # 检查必要的字段是否存在
#     description = parsed.get("description", "")
#     analysis = parsed.get("analysis", "")
#     answer = parsed.get("answer", "")
    
#     # 对于正常样本，需要description, analysis, answer
#     # 对于异常样本，还需要location
#     is_normal = sample.get("is_normal", True)
    
#     if is_normal:
#         if not description or not analysis or not answer:
#             return True
#     else:
#         location = parsed.get("location", "")
#         if not description or not analysis or not answer or not location:
#             return True
    
#     return False


# def create_sample_info(category: str, sample: Dict[str, Any], is_normal: bool) -> Dict[str, Any]:
#     """创建失败样本的详细信息"""
    
#     parsed = sample.get("parsed", {})
#     response = sample.get("response", "")
    
#     # 确定失败原因
#     error_reason = determine_error_reason(sample)
    
#     sample_info = {
#         "category": category,
#         "image_path": sample.get("image_path", "unknown"),
#         "is_normal": is_normal,
#         "defect_type": sample.get("defect_type", "N/A") if not is_normal else "N/A",
#         "raw_response": response,
#         "parsed_fields": {
#             "description": parsed.get("description", ""),
#             "analysis": parsed.get("analysis", ""),
#             "location": parsed.get("location", ""),
#             "answer": parsed.get("answer", "")
#         },
#         "parsing_success": parsed.get("parsing_success", False),
#         "response_status": "error" if response.lower() in ["error", "image loading error"] else "received",
#         "error_reason": error_reason,
#         "full_parsed_object": parsed  # 包含完整的解析对象以供进一步分析
#     }
    
#     return sample_info


# def determine_error_reason(sample: Dict[str, Any]) -> str:
#     """确定失败的具体原因"""
    
#     response = sample.get("response", "").lower()
#     parsed = sample.get("parsed", {})
    
#     # 检查响应
#     if response in ["error", "image loading error"]:
#         return f"响应错误: {response}"
    
#     if not response or response.strip() == "":
#         return "空响应"
    
#     # 检查解析成功标志
#     if not parsed.get("parsing_success", False):
#         return "parsing_success为False"
    
#     # 检查必要的字段
#     description = parsed.get("description", "")
#     analysis = parsed.get("analysis", "")
#     answer = parsed.get("answer", "")
#     location = parsed.get("location", "")
    
#     missing_fields = []
    
#     if not description:
#         missing_fields.append("description")
#     if not analysis:
#         missing_fields.append("analysis")
#     if not answer:
#         missing_fields.append("answer")
    
#     is_normal = sample.get("is_normal", True)
#     if not is_normal and not location:
#         missing_fields.append("location")
    
#     if missing_fields:
#         return f"缺少必要字段: {', '.join(missing_fields)}"
    
#     # 检查格式是否正确（是否包含必要的标签）
#     response_text = sample.get("response", "")
#     if is_normal:
#         if "<description>" not in response_text or "</description>" not in response_text:
#             return "缺少description标签"
#         if "<analysis>" not in response_text or "</analysis>" not in response_text:
#             return "缺少analysis标签"
#         if "<answer>" not in response_text or "</answer>" not in response_text:
#             return "缺少answer标签"
#     else:
#         if "<description>" not in response_text or "</description>" not in response_text:
#             return "缺少description标签"
#         if "<analysis>" not in response_text or "</analysis>" not in response_text:
#             return "缺少analysis标签"
#         if "<location>" not in response_text or "</location>" not in response_text:
#             return "缺少location标签"
#         if "<answer>" not in response_text or "</answer>" not in response_text:
#             return "缺少answer标签"
    
#     return "未知原因"


# def update_error_stats(stats: Dict[str, Any], sample: Dict[str, Any]):
#     """更新错误统计信息"""
    
#     parsed = sample.get("parsed", {})
#     response = sample.get("response", "").lower()
    
#     if not parsed.get("parsing_success", False):
#         stats["failed_by_parsing_success"] += 1
    
#     if response in ["error", "image loading error"]:
#         stats["failed_by_error_response"] += 1


# def analyze_failed_patterns(failed_samples_file: str):
#     """分析失败样本的模式"""
    
#     print("\n" + "="*60)
#     print("失败样本模式分析")
#     print("="*60)
    
#     try:
#         with open(failed_samples_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"错误: 文件 {failed_samples_file} 不存在")
#         return
    
#     failed_samples = data.get("failed_samples", [])
    
#     if not failed_samples:
#         print("没有失败样本可供分析")
#         return
    
#     # 按失败原因分组
#     error_reasons = {}
#     for sample in failed_samples:
#         reason = sample.get("error_reason", "未知原因")
#         error_reasons[reason] = error_reasons.get(reason, 0) + 1
    
#     print("\n失败原因分布:")
#     for reason, count in sorted(error_reasons.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {reason}: {count} 个样本 ({count/len(failed_samples)*100:.1f}%)")
    
#     # 按响应状态分组
#     response_statuses = {}
#     for sample in failed_samples:
#         status = sample.get("response_status", "unknown")
#         response_statuses[status] = response_statuses.get(status, 0) + 1
    
#     print("\n响应状态分布:")
#     for status, count in response_statuses.items():
#         print(f"  {status}: {count} 个样本")
    
#     # 检查特定错误模式
#     print("\n特定错误模式检查:")
    
#     # 检查空响应
#     empty_responses = [s for s in failed_samples if not s.get("raw_response", "").strip()]
#     if empty_responses:
#         print(f"  • 空响应: {len(empty_responses)} 个样本")
    
#     # 检查包含错误关键词的响应
#     error_keywords = ["error", "fail", "sorry", "cannot", "unable"]
#     keyword_errors = []
#     for sample in failed_samples:
#         response = sample.get("raw_response", "").lower()
#         for keyword in error_keywords:
#             if keyword in response:
#                 keyword_errors.append(sample)
#                 break
    
#     if keyword_errors:
#         print(f"  • 包含错误关键词: {len(keyword_errors)} 个样本")
#         # 显示一些示例
#         print("    示例响应:")
#         for i, sample in enumerate(keyword_errors[:3]):
#             print(f"      {i+1}. {sample.get('raw_response', '')[:100]}...")
    
#     # 生成修复建议
#     print("\n修复建议:")
#     if "缺少必要字段" in error_reasons:
#         print("  1. 检查模型输出格式，确保遵循指定的XML标签格式")
#         print("  2. 考虑调整提示词，明确要求必须包含所有字段")
    
#     if "parsing_success为False" in error_reasons:
#         print("  3. 检查解析逻辑，可能需要增强解析器的容错性")
    
#     if "响应错误" in error_reasons:
#         print("  4. 检查图像加载和模型推理过程，确保输入有效")


# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description="提取JSON文件中的解析失败样本")
#     parser.add_argument("--input", type=str,  default="32b_mvtec.json",
#                         help="输入JSON文件路径")
#     parser.add_argument("--output", type=str, default="failed_samples.json",
#                         help="输出JSON文件路径")
#     parser.add_argument("--analyze", action="store_true",
#                         help="分析失败样本的模式")
    
#     args = parser.parse_args()
    
#     # 提取失败样本
#     extract_failed_samples(args.input, args.output)
    
#     # 如果需要，分析失败模式
#     if args.analyze:
#         analyze_failed_patterns(args.output)


# if __name__ == "__main__":
#     main()