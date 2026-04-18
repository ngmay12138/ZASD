"""基于平衡准确率（Balanced Accuracy）的异常检测评估脚本。

评估指标：
    BAcc_per_category = (normal_acc + abnormal_acc) / 2
    Macro BAcc        = mean(BAcc_per_category)       各类别 BAcc 的宏平均
    Global BAcc       = (global_normal_acc + global_abnormal_acc) / 2  全局汇总
"""
import os
import json
import torch
import glob
from PIL import Image
from tqdm import tqdm
import argparse
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class AnomalyDetector:
    def __init__(self, model_path, batch_size=4, tensor_parallel_size=1):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("正在初始化 vLLM 引擎...")
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
            stop_token_ids=[self.tokenizer.eos_token_id],
        )

        self.prompt = (
            "Is there any abnormality in the object in this picture? "
            "If there is an abnormality, answer 'abnormal'; otherwise, answer 'normal'."
        )
        print("vLLM 引擎初始化完成！")

    # ------------------------------------------------------------------ #
    #  提示词 & 推理
    # ------------------------------------------------------------------ #
    def build_prompt(self) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_response(self, response: str) -> int:
        """解析模型回复，返回 1=异常 / 0=正常"""
        try:
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                answer = response[start:end].strip().lower()
                if "abnormal" in answer:
                    return 1
                if "normal" in answer:
                    return 0

            lower = response.lower()
            if any(w in lower for w in ["abnormal", "yes", "异常"]):
                return 1
            if any(w in lower for w in ["normal", "no", "正常"]):
                return 0
        except Exception as e:
            print(f"解析响应时出错: {e}")
        return 0

    def batch_predict(self, image_paths: List[str]) -> List[tuple]:
        if not image_paths:
            return []

        batch_inputs = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                batch_inputs.append({
                    "prompt": self.build_prompt(),
                    "multi_modal_data": {"image": [image]},
                })
            except Exception as e:
                print(f"加载图片 {image_path} 时出错: {e}")
                batch_inputs.append({"prompt": "", "multi_modal_data": {"image": []}})

        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)

        results = []
        for output in outputs:
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                results.append((self._parse_response(response), response))
            else:
                results.append((0, "error"))
        return results

    # ------------------------------------------------------------------ #
    #  数据集处理 & BAcc 计算
    # ------------------------------------------------------------------ #
    def process_dataset(
        self,
        dataset_path: str,
        meta_file_path: str,
        output_file: str = "results.json",
    ):
        print(f"开始处理数据集: {dataset_path}")

        with open(meta_file_path, "r") as f:
            meta_data = json.load(f)

        # 用 train 或 test 的 key 作为类别列表
        if meta_data.get("train"):
            categories = list(meta_data["train"].keys())
        else:
            categories = list(meta_data["test"].keys())
        print(f"找到的类别: {categories}")

        # ---------- 存储结构 ----------
        results: Dict = {
            "test": {},
            "statistics": {
                "total_images": 0,
                "correct_predictions": 0,
                "macro_bacc": 0.0,
                "global_bacc": 0.0,
                "class_wise": {},
            },
        }

        for category in categories:
            print(f"\n处理类别: {category}")
            results["test"][category] = []

            test_base_path = os.path.join(dataset_path, category, "test")
            if not os.path.exists(test_base_path):
                print(f"警告: {test_base_path} 不存在，跳过")
                continue

            subdirs = [
                d
                for d in os.listdir(test_base_path)
                if os.path.isdir(os.path.join(test_base_path, d))
            ]

            # 类别级别的正常/异常统计
            cat_normal_total = 0
            cat_normal_correct = 0
            cat_abnormal_total = 0
            cat_abnormal_correct = 0

            for subdir in subdirs:
                subdir_path = os.path.join(test_base_path, subdir)
                image_files = []
                for ext in [
                    "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif",
                    "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIFF", "*.TIF",
                ]:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
                image_files = list(set(image_files))

                true_label = 0 if subdir == "good" else 1
                print(
                    f"  子目录 {subdir} (label={true_label})，图片数: {len(image_files)}"
                )
                if not image_files:
                    continue

                for batch_start in tqdm(
                    range(0, len(image_files), self.batch_size), desc=f"  {subdir}"
                ):
                    batch_files = image_files[batch_start : batch_start + self.batch_size]
                    batch_predictions = self.batch_predict(batch_files)

                    for img_file, (prediction, model_response) in zip(
                        batch_files, batch_predictions
                    ):
                        is_correct = prediction == true_label

                        results["test"][category].append(
                            {
                                "img_path": os.path.relpath(img_file, dataset_path),
                                "cls_name": category,
                                "specie_name": subdir,
                                "true_anomaly": true_label,
                                "pred_anomaly": prediction,
                                "model_response": model_response,
                                "correct": is_correct,
                            }
                        )

                        results["statistics"]["total_images"] += 1
                        if is_correct:
                            results["statistics"]["correct_predictions"] += 1

                        if true_label == 0:
                            cat_normal_total += 1
                            if is_correct:
                                cat_normal_correct += 1
                        else:
                            cat_abnormal_total += 1
                            if is_correct:
                                cat_abnormal_correct += 1

            # ---------- 类别级 BAcc ----------
            normal_acc = (
                cat_normal_correct / cat_normal_total if cat_normal_total > 0 else 0.0
            )
            abnormal_acc = (
                cat_abnormal_correct / cat_abnormal_total
                if cat_abnormal_total > 0
                else 0.0
            )

            if cat_normal_total > 0 and cat_abnormal_total > 0:
                cat_bacc = (normal_acc + abnormal_acc) / 2
            elif cat_normal_total > 0 or cat_abnormal_total > 0:
                # 只有一种类别时退化为该类别的准确率
                cat_bacc = normal_acc if cat_normal_total > 0 else abnormal_acc
            else:
                cat_bacc = 0.0

            cat_total = cat_normal_total + cat_abnormal_total
            cat_correct = cat_normal_correct + cat_abnormal_correct
            cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0

            results["statistics"]["class_wise"][category] = {
                "total": cat_total,
                "correct": cat_correct,
                "accuracy": cat_acc,
                "normal_total": cat_normal_total,
                "normal_correct": cat_normal_correct,
                "normal_accuracy": normal_acc,
                "abnormal_total": cat_abnormal_total,
                "abnormal_correct": cat_abnormal_correct,
                "abnormal_accuracy": abnormal_acc,
                "balanced_accuracy": cat_bacc,
            }
            print(
                f"  {category}: Acc={cat_acc:.4f}  "
                f"NormalAcc={normal_acc:.4f}({cat_normal_correct}/{cat_normal_total})  "
                f"AbnormalAcc={abnormal_acc:.4f}({cat_abnormal_correct}/{cat_abnormal_total})  "
                f"BAcc={cat_bacc:.4f}"
            )

        # ---------- 总体指标 ----------
        total = results["statistics"]["total_images"]
        correct = results["statistics"]["correct_predictions"]

        # Macro BAcc（各类别 BAcc 的均值）
        bacc_values = [
            v["balanced_accuracy"]
            for v in results["statistics"]["class_wise"].values()
            if v["total"] > 0
        ]
        macro_bacc = sum(bacc_values) / len(bacc_values) if bacc_values else 0.0
        results["statistics"]["macro_bacc"] = macro_bacc

        # Global BAcc（所有类别的 normal/abnormal 汇总后计算）
        global_normal_total = sum(
            v["normal_total"] for v in results["statistics"]["class_wise"].values()
        )
        global_normal_correct = sum(
            v["normal_correct"] for v in results["statistics"]["class_wise"].values()
        )
        global_abnormal_total = sum(
            v["abnormal_total"] for v in results["statistics"]["class_wise"].values()
        )
        global_abnormal_correct = sum(
            v["abnormal_correct"] for v in results["statistics"]["class_wise"].values()
        )
        global_normal_acc = (
            global_normal_correct / global_normal_total if global_normal_total > 0 else 0.0
        )
        global_abnormal_acc = (
            global_abnormal_correct / global_abnormal_total
            if global_abnormal_total > 0
            else 0.0
        )
        global_bacc = (global_normal_acc + global_abnormal_acc) / 2
        results["statistics"]["global_bacc"] = global_bacc

        # ---------- 打印汇总 ----------
        print(f"\n{'=' * 60}")
        print(f"Macro BAcc:  {macro_bacc:.4f}  (各类别 BAcc 均值)")
        print(f"Global BAcc: {global_bacc:.4f}  (全局 NormalAcc={global_normal_acc:.4f}, AbnormalAcc={global_abnormal_acc:.4f})")
        print(f"{'=' * 60}")

        print("\n各类别平衡准确率:")
        for category, stats in results["statistics"]["class_wise"].items():
            print(
                f"  {category:25s}  BAcc={stats['balanced_accuracy']:.4f}  "
                f"Acc={stats['accuracy']:.4f}  "
                f"Normal={stats['normal_correct']}/{stats['normal_total']}  "
                f"Abnormal={stats['abnormal_correct']}/{stats['abnormal_total']}"
            )

        # ---------- 保存 ----------
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

        return results, macro_bacc

    # ------------------------------------------------------------------ #
    #  报告生成
    # ------------------------------------------------------------------ #
    def generate_report(self, results: Dict, report_file: str):
        os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
        stats = results["statistics"]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("Anomaly Detection Evaluation Report (Balanced Accuracy)\n")
            f.write("=" * 70 + "\n\n")

            f.write("Overall:\n")
            f.write(f"  Total images:              {stats['total_images']}\n")
            f.write(f"  Correct predictions:       {stats['correct_predictions']}\n")
            f.write(f"  Macro BAcc:                {stats['macro_bacc']:.4f}\n")
            f.write(f"  Global BAcc:               {stats['global_bacc']:.4f}\n\n")

            f.write(f"{'Category':25s}  {'BAcc':>8s}  {'Acc':>8s}  "
                    f"{'NormAcc':>8s}  {'AbnoAcc':>8s}  "
                    f"{'Norm':>10s}  {'Abno':>10s}\n")
            f.write("-" * 90 + "\n")

            for category, cs in stats["class_wise"].items():
                f.write(
                    f"{category:25s}  {cs['balanced_accuracy']:8.4f}  {cs['accuracy']:8.4f}  "
                    f"{cs['normal_accuracy']:8.4f}  {cs['abnormal_accuracy']:8.4f}  "
                    f"{cs['normal_correct']:>4d}/{cs['normal_total']:<5d}  "
                    f"{cs['abnormal_correct']:>4d}/{cs['abnormal_total']:<5d}\n"
                )

            f.write("-" * 90 + "\n")
            f.write(
                f"{'Macro Average':25s}  {stats['macro_bacc']:8.4f}\n"
            )

            # 错误案例
            f.write("\n\nError Cases (first 20):\n")
            error_cases = []
            for category in results["test"]:
                for item in results["test"][category]:
                    if not item["correct"]:
                        error_cases.append(item)

            if error_cases:
                f.write(f"  Total errors: {len(error_cases)}\n")
                for i, case in enumerate(error_cases[:20]):
                    f.write(f"  {i+1:3d}. {case['img_path']}\n")
                    f.write(
                        f"       true={case['true_anomaly']}  pred={case['pred_anomaly']}  "
                        f"response: {case['model_response'][:120]}\n"
                    )
            else:
                f.write("  None!\n")

        print(f"详细报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="异常检测评估（平衡准确率 BAcc）"
    )
    parser.add_argument("--model-path", type=str, default="ch_trl/my_model/qwen3_samsftrl_6k/checkpoint-3324")
    parser.add_argument("--dataset-path", type=str, default="ch_llamafactory/LlamaFactory-main/data/mvtec_ad")
    parser.add_argument("--meta-file", type=str, default="ch_llamafactory/LlamaFactory-main/data/mvtec_ad/meta.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument(
        "--output-file", type=str, default="ch_llamafactory/LlamaFactory-main/results/bacc_check1504/mvtec_ad.json"
    )
    parser.add_argument(
        "--report-file", type=str, default="ch_llamafactory/LlamaFactory-main/results/bacc_check1504/mvtec_ad.txt"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在！")
        return
    if not os.path.exists(args.meta_file):
        print(f"错误: meta.json 文件 {args.meta_file} 不存在！")
        return

    print(f"使用 vLLM 加载模型: {args.model_path}")
    detector = AnomalyDetector(
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    results, macro_bacc = detector.process_dataset(
        dataset_path=args.dataset_path,
        meta_file_path=args.meta_file,
        output_file=args.output_file,
    )

    detector.generate_report(results=results, report_file=args.report_file)


if __name__ == "__main__":
    main()
