import os
import json
import base64
import glob
import asyncio
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
import io
import time

import anthropic


class AnomalyDetector:
    def __init__(self, model="claude-opus-4-6", max_concurrent=8, enable_thinking=True):
        """
        初始化异常检测器 (Claude API 版本)

        Args:
            model: Claude 模型名称
            max_concurrent: 最大并发请求数
            enable_thinking: 是否启用扩展思考 (effort=max)
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self.enable_thinking = enable_thinking

        base_url = os.environ.get("ANTHROPIC_BASE_URL", "http://10.162.51.138:8080")
        api_key = os.environ.get(
            "ANTHROPIC_AUTH_TOKEN",
            "sk-d02222b704f8ec17de68eabd6a68fc0d5f54cb8385166de836432a8f2b1d015a",
        )

        self.client = anthropic.AsyncAnthropic(base_url=base_url, api_key=api_key)

        # 与原始 infer.py 完全相同的 prompt，但增加结构化输出要求
        self.prompt = (
            "Is there any abnormality in the object in this picture? "
            "If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'. "
            "You must wrap your final answer in <answer> tags, for example: <answer>normal</answer> or <answer>abnormal</answer>."
        )

        print(f"模型: {self.model}")
        print(f"API 地址: {base_url}")
        print(f"最大并发: {self.max_concurrent}")
        print(f"扩展思考: {'开启' if self.enable_thinking else '关闭'}")

    def encode_image(self, image_path: str) -> Tuple[str, str]:
        """加载图片并编码为 base64，与原始 infer.py 一致直接发送原图"""
        image = Image.open(image_path).convert("RGB")

        ext = Path(image_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            fmt, media_type = "JPEG", "image/jpeg"
        else:
            fmt, media_type = "PNG", "image/png"

        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), media_type

    async def predict_single(
        self, image_path: str, semaphore: asyncio.Semaphore, max_retries: int = 3
    ) -> Tuple[int, str]:
        """单张图片推理，带重试"""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    b64_data, media_type = self.encode_image(image_path)

                    kwargs = dict(
                        model=self.model,
                        max_tokens=16000,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64_data,
                                        },
                                    },
                                    {"type": "text", "text": self.prompt},
                                ],
                            }
                        ],
                    )

                    if self.enable_thinking:
                        kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 10000,
                        }
                    else:
                        kwargs["temperature"] = 0.0

                    response = await self.client.messages.create(**kwargs)

                    # 提取文本响应 (跳过 thinking blocks)
                    text_response = ""
                    for block in response.content:
                        if block.type == "text":
                            text_response += block.text

                    prediction = self._parse_response(text_response)
                    return prediction, text_response

                except anthropic.RateLimitError:
                    wait = 2 ** (attempt + 1)
                    print(f"  速率限制，{wait}s 后重试: {Path(image_path).name}")
                    await asyncio.sleep(wait)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"  处理失败 {Path(image_path).name}: {e}")
                        return 0, f"error: {e}"
                    await asyncio.sleep(1)

            return 0, "error: max retries exceeded"

    def _parse_response(self, response: str) -> int:
        """解析模型响应，优先使用 <answer> 标签，fallback 使用首行/首词匹配"""
        prediction = 0
        try:
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                answer_content = response[start:end].strip().lower()
                prediction = 1 if "abnormal" in answer_content else 0
            else:
                # fallback: 提取响应的第一个有意义的词进行判断
                # 避免在长文本中误匹配（如 "no abnormalities" 被匹配为 abnormal）
                import re
                resp_lower = response.strip().lower()
                # 去除 markdown 加粗符号后取第一个词
                first_word = re.sub(r'[*_#`>]', '', resp_lower).strip().split()[0] if resp_lower else ""
                if first_word in ("abnormal", "abnormal.", "abnormal,", "abnormal:"):
                    prediction = 1
                elif first_word in ("normal", "normal.", "normal,", "normal:"):
                    prediction = 0
                # 如果首词无法判断，再用全文关键词匹配（但用完整词匹配）
                elif re.search(r'\babnormal\b', resp_lower):
                    prediction = 1
                elif re.search(r'\bnormal\b', resp_lower):
                    prediction = 0
                else:
                    print(f"  警告: 无法解析响应，默认为 normal: {resp_lower[:100]}")
        except Exception as e:
            print(f"  解析响应出错: {e}")
        return prediction

    async def batch_predict(self, image_paths: List[str]) -> List[Tuple[int, str]]:
        """批量并发推理"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self.predict_single(p, semaphore) for p in image_paths]
        return await asyncio.gather(*tasks)

    def process_dataset(self, dataset_path, meta_file_path, output_file="results.json"):
        """处理数据集，逻辑与原始 infer.py 保持一致"""
        print(f"开始处理数据集: {dataset_path}")

        with open(meta_file_path, "r") as f:
            meta_data = json.load(f)

        results = {
            "test": {},
            "statistics": {
                "total_images": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "class_wise_accuracy": {},
            },
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

            subdirs = [
                d
                for d in os.listdir(test_base_path)
                if os.path.isdir(os.path.join(test_base_path, d))
            ]

            total_category = 0
            correct_category = 0

            for subdir in subdirs:
                subdir_path = os.path.join(test_base_path, subdir)

                image_files = []
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif",
                            "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIFF", "*.TIF"]:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
                image_files = list(set(image_files))

                true_label = 0 if subdir == "good" else 1

                print(f"  子目录 {subdir} (标签: {true_label})，图片数: {len(image_files)}")
                if not image_files:
                    continue

                # 按批次并发处理
                for batch_start in tqdm(
                    range(0, len(image_files), self.max_concurrent),
                    desc=f"  {subdir}",
                ):
                    batch_files = image_files[batch_start : batch_start + self.max_concurrent]
                    batch_results = asyncio.run(self.batch_predict(batch_files))

                    for img_file, (prediction, model_response) in zip(batch_files, batch_results):
                        is_correct = prediction == true_label
                        results["test"][category].append(
                            {
                                "img_path": os.path.relpath(img_file, dataset_path),
                                "mask_path": "",
                                "cls_name": category,
                                "specie_name": subdir,
                                "true_anomaly": true_label,
                                "pred_anomaly": prediction,
                                "model_response": model_response,
                                "correct": is_correct,
                            }
                        )
                        total_category += 1
                        results["statistics"]["total_images"] += 1
                        if is_correct:
                            correct_category += 1
                            results["statistics"]["correct_predictions"] += 1

            if total_category > 0:
                acc = correct_category / total_category
                results["statistics"]["class_wise_accuracy"][category] = {
                    "total": total_category,
                    "correct": correct_category,
                    "accuracy": acc,
                }
                print(f"  类别 {category} 准确率: {acc:.4f} ({correct_category}/{total_category})")

        if results["statistics"]["total_images"] > 0:
            overall_accuracy = (
                results["statistics"]["correct_predictions"]
                / results["statistics"]["total_images"]
            )
            results["statistics"]["accuracy"] = overall_accuracy

            print(f"\n{'='*50}")
            print(f"总体准确率: {overall_accuracy:.4f}")
            print(
                f"正确预测: {results['statistics']['correct_predictions']}"
                f"/{results['statistics']['total_images']}"
            )
            print(f"{'='*50}")
            print("\n各类别准确率:")
            for cat, stats in results["statistics"]["class_wise_accuracy"].items():
                print(f"  {cat}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

        return results, results["statistics"].get("accuracy", 0.0)

    def generate_report(self, results, report_file="anomaly_detection_report.txt"):
        """生成详细报告，与原始 infer.py 逻辑一致"""
        os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("=" * 60 + "\n\n")

            stats = results["statistics"]
            f.write(f"总体统计:\n")
            f.write(f"  总图片数: {stats['total_images']}\n")
            f.write(f"  正确预测数: {stats['correct_predictions']}\n")
            f.write(f"  准确率: {stats['accuracy']:.4f}\n\n")

            f.write("各类别统计:\n")
            for category, cs in stats["class_wise_accuracy"].items():
                f.write(f"  {category}:\n")
                f.write(f"    图片数: {cs['total']}\n")
                f.write(f"    正确数: {cs['correct']}\n")
                f.write(f"    准确率: {cs['accuracy']:.4f}\n\n")

            f.write("错误案例分析:\n")
            error_cases = [
                item
                for cat in results["test"]
                for item in results["test"][cat]
                if not item["correct"]
            ]
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
    parser = argparse.ArgumentParser(description="使用 Claude API 的异常检测")
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--dataset-path", type=str, default="data/visa")
    parser.add_argument("--meta-file", type=str, default="data/visa/meta.json")
    parser.add_argument("--max-concurrent", type=int, default=8, help="最大并发请求数")
    parser.add_argument("--output-file", type=str, default="results/claude_test.json")
    parser.add_argument("--report-file", type=str, default="results/claude_test.txt")
    parser.add_argument("--no-thinking", action="store_true", help="关闭扩展思考")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return
    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json 文件 {args.meta_file} 不存在！")
        return

    detector = AnomalyDetector(
        model=args.model,
        max_concurrent=args.max_concurrent,
        enable_thinking=not args.no_thinking,
    )

    results, accuracy = detector.process_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file,
    )
    detector.generate_report(results=results, report_file=args.report_file)


if __name__ == "__main__":
    main()
