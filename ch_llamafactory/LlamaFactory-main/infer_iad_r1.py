import os
import json
import torch
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class AnomalyDetector:
    def __init__(self, model_path, device=None, batch_size=4, tensor_parallel_size=1):
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"使用设备: {self.device}")
        print(f"批处理大小: {batch_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("正在初始化vLLM引擎...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16" if self.device == "cuda" else "float32",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=1000,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        self.prompt = """Is there any abnormality in the object in this picture? If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."""
        print("vLLM引擎初始化完成！")

    def build_prompt(self) -> str:
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
        if not image_paths:
            return []

        batch_inputs = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                prompt = self.build_prompt()
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": [image]}
                })
            except Exception as e:
                print(f"加载图片 {image_path} 时出错: {e}")
                batch_inputs.append({
                    "prompt": "",
                    "multi_modal_data": {"image": []}
                })

        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)

        results = []
        for output in outputs:
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                prediction = self._parse_response(response)
                results.append((prediction, response))
            else:
                results.append((0, "error"))

        return results

    def _parse_response(self, response: str) -> int:
        """解析模型回复。iad-r1 回答 <answer>Yes</answer>(异常) / <answer>No</answer>(正常)"""
        try:
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                answer = response[start:end].strip().lower()

                if "yes" in answer:
                    return 1
                if "no" in answer:
                    return 0

            # 后备：无 <answer> 标签时按关键词匹配
            lower = response.lower()
            if any(w in lower for w in ["yes", "abnormal", "异常"]):
                return 1
            if any(w in lower for w in ["no", "normal", "正常"]):
                return 0
        except Exception as e:
            print(f"解析响应时出错: {e}")

        return 0

    def process_dataset(self, dataset_path, meta_file_path, output_file="results.json"):
        print(f"开始处理数据集: {dataset_path}")

        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)

        results = {
            "test": {},
            "statistics": {
                "total_images": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "class_wise_accuracy": {}
            }
        }

        categories = list(meta_data["train"].keys())
        print(f"找到的类别: {categories}")

        for category in categories:
            print(f"\n处理类别: {category}")
            results["test"][category] = []

            test_base_path = os.path.join(dataset_path, category, "test")
            if not os.path.exists(test_base_path):
                print(f"警告: {test_base_path} 不存在，跳过此类别")
                continue

            subdirs = [d for d in os.listdir(test_base_path)
                       if os.path.isdir(os.path.join(test_base_path, d))]

            total_category = 0
            correct_category = 0

            for subdir in subdirs:
                subdir_path = os.path.join(test_base_path, subdir)

                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif',
                            '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF', '*.TIF']:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
                image_files = list(set(image_files))

                true_label = 0 if subdir == "good" else 1

                print(f"  处理子目录 {subdir} (真实标签: {true_label})，图片数: {len(image_files)}")

                if not image_files:
                    print(f"  警告: 子目录 {subdir} 中没有找到支持的图片文件")
                    continue

                for batch_start in tqdm(range(0, len(image_files), self.batch_size),
                                        desc=f"  {subdir}"):
                    batch_files = image_files[batch_start:batch_start + self.batch_size]
                    batch_predictions = self.batch_predict(batch_files)

                    for img_file, (prediction, model_response) in zip(batch_files, batch_predictions):
                        is_correct = (prediction == true_label)

                        results["test"][category].append({
                            "img_path": os.path.relpath(img_file, dataset_path),
                            "mask_path": "",
                            "cls_name": category,
                            "specie_name": subdir,
                            "true_anomaly": true_label,
                            "pred_anomaly": prediction,
                            "model_response": model_response,
                            "correct": is_correct
                        })

                        total_category += 1
                        results["statistics"]["total_images"] += 1

                        if is_correct:
                            correct_category += 1
                            results["statistics"]["correct_predictions"] += 1

            if total_category > 0:
                category_accuracy = correct_category / total_category
                results["statistics"]["class_wise_accuracy"][category] = {
                    "total": total_category,
                    "correct": correct_category,
                    "accuracy": category_accuracy
                }
                print(f"  类别 {category} 准确率: {category_accuracy:.4f} ({correct_category}/{total_category})")

        if results["statistics"]["total_images"] > 0:
            overall_accuracy = results["statistics"]["correct_predictions"] / results["statistics"]["total_images"]
            results["statistics"]["accuracy"] = overall_accuracy

            print(f"\n{'='*50}")
            print(f"总体准确率: {overall_accuracy:.4f}")
            print(f"正确预测: {results['statistics']['correct_predictions']}/{results['statistics']['total_images']}")
            print(f"{'='*50}")

            print("\n各类别准确率:")
            for category, stats in results["statistics"]["class_wise_accuracy"].items():
                print(f"  {category}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        else:
            overall_accuracy = 0.0

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

        return results, overall_accuracy

    def generate_report(self, results, report_file="anomaly_detection_report.txt"):
        os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("iad-r1 (Qwen2-VL-2B) Anomaly Detection Report\n")
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
    parser = argparse.ArgumentParser(description="iad-r1 (Qwen2-VL-2B) 异常检测测试")
    parser.add_argument("--model-path", type=str, default="model/iad-r1(qwen2-2b)")
    parser.add_argument("--dataset-path", type=str, default="data/dagm")
    parser.add_argument("--meta-file", type=str, default="data/dagm/meta.json")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批处理大小")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPU并行数量")
    parser.add_argument("--output-file", type=str, default="results/iad_r1_test.json",
                        help="输出结果文件")
    parser.add_argument("--report-file", type=str, default="results/iad_r1_test.txt",
                        help="报告文件")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return

    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json文件 {args.meta_file} 不存在！")
        return

    print(f"使用vLLM加载模型: {args.model_path}")
    detector = AnomalyDetector(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )

    results, accuracy = detector.process_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file
    )

    detector.generate_report(
        results=results,
        report_file=args.report_file
    )


if __name__ == "__main__":
    main()
