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

class AnomalyDetector:
    def __init__(self, model_path, device=None, batch_size=4, tensor_parallel_size=1):
        """
        初始化异常检测器
        
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
            temperature=0.0,
            top_p=0.9,
            max_tokens=1000,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        # 固定的提示词
#         self.prompt = """ Please carefully analyze the objects in this image to determine if there is an anomaly.If an abnormality exists, answer abnormal; otherwise, answer normal.
# ## Task Description
# You are an anomaly detection expert and need to identify defects, damage, abnormal states, or non-compliance issues in objects in images.
# ## Output Format Requirements
# If there are no abnormalities, answer strictly in the following format. Replace the description and analysis with your response, and do not add any additional text:
# <description>Briefly describe the key features you observe</description><analysis>Analysis of why the observed object is normal</analysis><answer>normal</answer>
# If there is an abnormality, answer strictly in the following format. Replace the description, analysis, and location with your response, and do not add any additional text:
# <description>Briefly describe the key features you observe</description><analysis>"Analysis of why the observed object is an anomaly"</analysis><location>The specific location of the anomaly</location><answer>abnormal</answer>
# Now analyze this image:"""
        # self.prompt = """Is there any abnormality in the object in this picture? If there is an abnormality, answer '<answer>abnormal</answer>'; otherwise, answer '<answer>normal</answer>'."""
        self.prompt = """Is there any abnormality in the object in this picture? If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."""
        print("vLLM引擎初始化完成！")
    
    def build_prompt(self) -> str:
        """构建提示词"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def batch_predict(self, image_paths: List[str]) -> List[tuple]:
        """
        批量预测图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            predictions: 预测结果列表，每个元素为(prediction, response)
        """
        if not image_paths:
            return []
        
        # 构建批次
        all_prompts = []
        all_images = []
        
        for image_path in image_paths:
            try:
                # 加载图片 - 支持多种格式
                image = Image.open(image_path).convert("RGB")
                
                # 构建提示词
                prompt = self.build_prompt()
                
                all_prompts.append(prompt)
                all_images.append([image])  # vLLM期望图片列表
            except Exception as e:
                print(f"加载图片 {image_path} 时出错: {e}")
                # 添加空条目作为占位符
                all_prompts.append("")
                all_images.append([])
        
        # 准备批次输入
        batch_inputs = []
        for prompt, images in zip(all_prompts, all_images):
            if prompt:  # 只处理有效输入
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": images}
                })
            else:
                # 对于无效输入，添加空输入
                batch_inputs.append({
                    "prompt": "",
                    "multi_modal_data": {"image": []}
                })
        
        # 批量推理
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        # 解析结果 - 修改为从<answer>字段中判断
        results = []
        for i, output in enumerate(outputs):
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                
                # 新的解析逻辑：从<answer>字段中判断是否存在"abnormal"
                prediction = 0  # 默认为正常
                try:
                    # 查找<answer>标签
                    if "<answer>" in response and "</answer>" in response:
                        start = response.find("<answer>") + len("<answer>")
                        end = response.find("</answer>", start)
                        answer_content = response[start:end].strip().lower()
                        
                        # 判断<answer>字段中是否包含"abnormal"
                        if "abnormal" in answer_content:
                            prediction = 1  # 异常
                        else:
                            prediction = 0  # 正常
                    else:
                        # 如果没有找到<answer>标签，尝试使用旧逻辑作为后备
                        if any(word in response.lower() for word in ["异常", "yes", "abnormal"]):
                            prediction = 1
                        elif any(word in response.lower() for word in ["正常", "no", "normal"]):
                            prediction = 0
                        else:
                            prediction = 0  # 默认值
                except Exception as e:
                    print(f"解析响应时出错: {e}")
                    prediction = 0
                        
                results.append((prediction, response))
            else:
                results.append((0, "error"))
        
        return results
    
    def process_dataset(self, dataset_path, meta_file_path, output_file="results.json"):
        print(f"开始处理数据集: {dataset_path}")
        
        # 加载meta.json文件
        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)
        
        # 存储结果
        results = {
            "test": {},
            "statistics": {
                "total_images": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "class_wise_accuracy": {}
            }
        }
        
        # 获取所有类别
        categories = list(meta_data["train"].keys())
        print(f"找到的类别: {categories}")
        
        # 遍历每个类别
        for category in categories:
            print(f"\n处理类别: {category}")
            
            # 初始化类别结果
            results["test"][category] = []
            
            # 构建test目录路径
            test_base_path = os.path.join(dataset_path, category, "test")
            
            if not os.path.exists(test_base_path):
                print(f"警告: {test_base_path} 不存在，跳过此类别")
                continue
            
            # 获取所有子目录（包括good和不同的缺陷类型）
            subdirs = [d for d in os.listdir(test_base_path) 
                      if os.path.isdir(os.path.join(test_base_path, d))]
            
            total_category = 0
            correct_category = 0
            
            for subdir in subdirs:
                subdir_path = os.path.join(test_base_path, subdir)
                
                # 支持多种图片格式
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
                
                # 还可以添加不区分大小写的扩展名
                for ext in ['*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF', '*.TIF']:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
                
                # 去重（如果同一个文件有多种扩展名的情况）
                image_files = list(set(image_files))
                
                # 确定真实标签：good为0，其他为1
                true_label = 0 if subdir == "good" else 1
                
                print(f"  处理子目录 {subdir} (真实标签: {true_label})，图片数: {len(image_files)}")
                
                if not image_files:
                    print(f"  警告: 子目录 {subdir} 中没有找到支持的图片文件")
                    continue
                
                # 按批次处理图片
                for batch_start in tqdm(range(0, len(image_files), self.batch_size), 
                                       desc=f"  {subdir}"):
                    batch_files = image_files[batch_start:batch_start + self.batch_size]
                    
                    # 批量预测
                    batch_predictions = self.batch_predict(batch_files)
                    
                    # 处理批量结果
                    for img_file, (prediction, model_response) in zip(batch_files, batch_predictions):
                        # 检查是否正确
                        is_correct = (prediction == true_label)
                        
                        # 记录结果
                        result_entry = {
                            "img_path": os.path.relpath(img_file, dataset_path),
                            "mask_path": "",
                            "cls_name": category,
                            "specie_name": subdir,
                            "true_anomaly": true_label,
                            "pred_anomaly": prediction,
                            "model_response": model_response,
                            "correct": is_correct
                        }
                        
                        results["test"][category].append(result_entry)
                        
                        # 更新统计
                        total_category += 1
                        results["statistics"]["total_images"] += 1
                        
                        if is_correct:
                            correct_category += 1
                            results["statistics"]["correct_predictions"] += 1
            
            # 计算类别准确率
            if total_category > 0:
                category_accuracy = correct_category / total_category
                results["statistics"]["class_wise_accuracy"][category] = {
                    "total": total_category,
                    "correct": correct_category,
                    "accuracy": category_accuracy
                }
                print(f"  类别 {category} 准确率: {category_accuracy:.4f} ({correct_category}/{total_category})")
        
        # 计算总体准确率
        if results["statistics"]["total_images"] > 0:
            overall_accuracy = results["statistics"]["correct_predictions"] / results["statistics"]["total_images"]
            results["statistics"]["accuracy"] = overall_accuracy
            
            print(f"\n{'='*50}")
            print(f"总体准确率: {overall_accuracy:.4f}")
            print(f"正确预测: {results['statistics']['correct_predictions']}/{results['statistics']['total_images']}")
            print(f"{'='*50}")
            
            # 打印每个类别的准确率
            print("\n各类别准确率:")
            for category, stats in results["statistics"]["class_wise_accuracy"].items():
                print(f"  {category}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        # 保存结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
        
        return results, overall_accuracy
    
    def generate_report(self, results, report_file="anomaly_detection_report.txt"):
        """
        生成详细的检测报告
        
        Args:
            results: 处理结果
            report_file: 报告文件路径
        """
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MVTec AD数据集异常检测报告\n")
            f.write("=" * 60 + "\n\n")
            
            stats = results["statistics"]
            f.write(f"总体统计:\n")
            f.write(f"  总图片数: {stats['total_images']}\n")
            f.write(f"  正确预测数: {stats['correct_predictions']}\n")
            f.write(f"  准确率: {stats['accuracy']:.4f}\n\n")
            
            f.write("各类别统计:\n")
            for category, category_stats in stats["class_wise_accuracy"].items():
                f.write(f"  {category}:\n")
                f.write(f"    图片数: {category_stats['total']}\n")
                f.write(f"    正确数: {category_stats['correct']}\n")
                f.write(f"    准确率: {category_stats['accuracy']:.4f}\n\n")
            
            # 分析错误案例
            f.write("错误案例分析:\n")
            error_cases = []
            for category in results["test"]:
                for item in results["test"][category]:
                    if not item["correct"]:
                        error_cases.append(item)
            
            if error_cases:
                f.write(f"  总错误数: {len(error_cases)}\n")
                f.write(f"  前10个错误案例:\n")
                for i, case in enumerate(error_cases[:10]):
                    f.write(f"    {i+1}. {case['img_path']}\n")
                    f.write(f"       真实标签: {case['true_anomaly']}, 预测标签: {case['pred_anomaly']}\n")
                    f.write(f"       模型回答: {case['model_response']}\n")
            else:
                f.write("  无错误案例！\n")
        
        print(f"详细报告已保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用vLLM加速的异常检测")
    parser.add_argument("--model-path", type=str, default="qwen3_sam4sft")
    parser.add_argument("--dataset-path", type=str, default="data/sdd")
    parser.add_argument("--meta-file", type=str, default="data/sdd/meta.json")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批处理大小")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="GPU并行数量")
    parser.add_argument("--output-file", type=str, default="results/test.json",
                        help="输出结果文件")
    parser.add_argument("--report-file", type=str, default="results/test.txt",
                        help="报告文件")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return
    
    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json文件 {args.meta_file} 不存在！")
        return
    
    # 创建检测器实例（使用vLLM）
    print(f"使用vLLM加载模型: {args.model_path}")
    detector = AnomalyDetector(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # 处理数据集
    results, accuracy = detector.process_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file
    )
    
    # 生成详细报告
    detector.generate_report(
        results=results,
        report_file=args.report_file
    )


if __name__ == "__main__":
    main()