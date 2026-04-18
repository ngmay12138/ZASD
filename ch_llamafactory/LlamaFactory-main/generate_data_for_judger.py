##################################构造句子对###################################
import os
import json
import random
import re
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image

# vLLM 相关导入
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ------------------------------
# 配置参数（可通过命令行覆盖）
# ------------------------------
DEFAULT_MODEL_PATH = "model/Qwen3-VL-32B"
DEFAULT_DATASET_ROOT = "data/Real-IAD"
DEFAULT_OUTPUT_FILE = "data4judger/sentence_pairs.jsonl"
DEFAULT_DESC_CACHE = "data4judger/image_descriptions_cache.json"
DEFAULT_BATCH_SIZE = 1
DEFAULT_TENSOR_PARALLEL_SIZE = 4
DEFAULT_TEMPERATURES = [0.0]          # 生成描述只使用温度0
DEFAULT_SEED = 42

# 新增：改写相关参数
DEFAULT_NUM_REWRITES = 1               # 每条描述每个温度生成多少个改写（此处每个温度只生成1个）
# 修改：改写温度默认列表（0.1到1.0步长0.1）
DEFAULT_REWRITE_TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_MAX_REWRITE_TOKENS = 256        # 改写生成的最大长度

# 困难负例关键词替换映射（遍历替换用）
KEYWORD_MAP = {
    #反义词替换
    "normal": ["abnormal", "anomalous"],
    "abnormal": ["normal", "good"],
    "good": ["bad", "abnormal"],
    "bad": ["good", "normal"],
    #异常类型替换
    "pit":["deformation","abrasion","scratch","damage","contamination"],
    "deformation":["pit","abrasion","scratch","damage","contamination"],
    "abrasion":["deformation","pit","scratch","damage","contamination"],
    "scratch":["deformation","abrasion","pit","damage","contamination"],
    "damage":["deformation","abrasion","scratch","pit","contamination"],
    "contamination":["deformation","abrasion","scratch","damage","pit"],
}


def traverse_replace(text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
    """
    对文本中的关键词进行遍历替换，每个关键词的每个候选词都生成一个替换后的文本。
    返回去重后的替换文本列表。
    """
    pattern = r'\b(' + '|'.join(re.escape(src) for src in keyword_map.keys()) + r')\b'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return []
    replaced_texts = set()
    for match in matches:
        src = match.group(0)
        tgts = keyword_map[src]
        for tgt in tgts:
            # 只替换当前匹配位置，其他关键词保持不变
            new_text = text[:match.start()] + tgt + text[match.end():]
            if new_text != text:  # 避免原词与候选相同（一般不会，但防意外）
                replaced_texts.add(new_text)
    return list(replaced_texts)


class QwenVLDescriptor:
    """
    使用 Qwen3-VL (vLLM) 为工业图像生成结构化描述，并支持纯文本改写。
    """
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        batch_size: int = 4,
        device: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        self.tensor_parallel_size = tensor_parallel_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"正在初始化 vLLM 引擎，模型路径: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16" if self.device == "cuda" else "float32",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},
        )
        print("vLLM 引擎初始化完成！")

    def _build_prompt(self, is_normal: bool) -> str:
        """根据图像类型构建提示词。正常样本单图输入，异常样本双图输入（原图+mask）。"""
        if is_normal:
            prompt = """
## Task Description
You are an industrial anomaly detection expert and need to explain why there is no abnormal situation in the objects in the image.
## Output Format Requirements
Answer strictly in the following format. Replace the think with your response, and do not add any additional text:
<think>Briefly describe your thought process about why the object in the image is normal in 1-2 sentences</think><answer>normal</answer>
Now analyze this image:"""
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
            prompt = """
## Task Description
You are an industrial anomaly detection expert. You are given two images: the first is the original object image, and the second is a defect mask image (highlighting the anomalous regions). Identify the defects, damage, abnormal states, or non-compliance issues in the object based on both images.
## Output Format Requirements
Answer strictly in the following format. Replace the description and think with your response, and do not add any additional text:
<description>A short phrase (3-8 words) describing the anomaly's appearance and location</description><think>Briefly describe your thought process about why the object in the image is abnormal in 1-2 sentences</think><answer>abnormal</answer>
Now analyze the images:"""
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

    def _parse_response(self, response: str) -> Dict[str, str]:
        """解析模型生成的文本，提取 description, think, answer"""
        result = {
            "description": "",
            "think": "",
            "answer": "",
            "parsing_success": False,
            "error": None
        }
        try:
            desc_start = response.find("<description>")
            desc_end = response.find("</description>")
            if desc_start != -1 and desc_end != -1 and desc_end > desc_start:
                result["description"] = response[desc_start + len("<description>"):desc_end].strip()
            think_start = response.find("<think>")
            think_end = response.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                result["think"] = response[think_start + len("<think>"):think_end].strip()
            answer_start = response.find("<answer>")
            answer_end = response.find("</answer>")
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                result["answer"] = response[answer_start + len("<answer>"):answer_end].strip()

            # think 和 answer 为必需字段；description 仅异常样本必需
            if result["think"] and result["answer"]:
                if result["answer"].lower() in ["normal", "abnormal"]:
                    result["parsing_success"] = True
                else:
                    result["error"] = f"无效 answer: {result['answer']}"
            else:
                missing = [f for f in ["think", "answer"] if not result[f]]
                result["error"] = f"缺失字段: {', '.join(missing)}"
        except Exception as e:
            result["error"] = str(e)
        return result

    def generate_batch(
        self,
        image_paths: List[str],
        is_normal_list: List[bool],
        mask_paths: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 1000,
    ) -> List[Dict]:
        """批量生成描述。异常样本使用双图输入（原图+mask）。"""
        assert len(image_paths) == len(is_normal_list)
        if mask_paths is None:
            mask_paths = [""] * len(image_paths)
        assert len(mask_paths) == len(image_paths)

        inputs = []
        valid_indices = []
        for i, (img_path, is_normal, mask_path) in enumerate(
            zip(image_paths, is_normal_list, mask_paths)
        ):
            try:
                image = Image.open(img_path).convert("RGB")
                prompt = self._build_prompt(is_normal)

                if is_normal or not mask_path:
                    # 正常样本：单图输入
                    inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": [image]}
                    })
                else:
                    # 异常样本：双图输入（原图 + mask）
                    mask_image = Image.open(mask_path).convert("RGB")
                    inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": [image, mask_image]}
                    })
                valid_indices.append(i)
            except Exception as e:
                print(f"加载图片 {img_path} 失败: {e}")

        if not inputs:
            return []

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        results = [None] * len(image_paths)
        for idx_in_batch, output in enumerate(outputs):
            original_idx = valid_indices[idx_in_batch]
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                parsed = self._parse_response(response)
                results[original_idx] = {
                    "image_path": image_paths[original_idx],
                    "is_normal": is_normal_list[original_idx],
                    "raw_response": response,
                    "parsed": parsed,
                    "temperature": temperature
                }
            else:
                results[original_idx] = {
                    "image_path": image_paths[original_idx],
                    "is_normal": is_normal_list[original_idx],
                    "raw_response": "",
                    "parsed": {"parsing_success": False, "error": "生成失败"},
                    "temperature": temperature
                }

        for i in range(len(image_paths)):
            if results[i] is None:
                results[i] = {
                    "image_path": image_paths[i],
                    "is_normal": is_normal_list[i],
                    "raw_response": "",
                    "parsed": {"parsing_success": False, "error": "图片加载失败"},
                    "temperature": temperature
                }
        return results

    def generate_multi_temp(
        self,
        image_paths: List[str],
        is_normal_list: List[bool],
        temperatures: List[float],
        mask_paths: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """对同一批图片使用多个 temperature 生成"""
        all_results = {}
        for temp in temperatures:
            print(f"使用 temperature={temp} 生成...")
            results = self.generate_batch(
                image_paths, is_normal_list,
                mask_paths=mask_paths, temperature=temp
            )
            all_results[temp] = results
        return all_results

    # ---------- 文本改写功能 ----------
    def rewrite_texts(
        self,
        texts: List[str],
        rewrite_type: str,          # "similar" 或 "opposite"
        temperature: float = 1.0,
        num_return: int = 1,
        max_tokens: int = 256,
        top_p: float = 0.9,
    ) -> List[List[str]]:
        """
        对文本列表进行改写，返回每个输入对应的改写结果列表（长度为 num_return）。

        rewrite_type:
            - "similar": 保持原意不变，换种说法
            - "opposite": 表达与原句相反的意思
        """
        if not texts:
            return []

        # 构建提示词
        if rewrite_type == "similar":
            instruction = "请用不同的表达方式改写以下句子，保持原意不变。直接输出改写后的句子，不要任何解释。\n\n句子："
        elif rewrite_type == "opposite":
            instruction = "请改写以下句子，使其表达与原句完全相反的意思，但尽量保持其他描述细节不变。直接输出改写后的句子，不要任何解释。\n\n句子："
        else:
            raise ValueError(f"未知 rewrite_type: {rewrite_type}")

        prompts = [instruction + t for t in texts]

        # 采样参数（启用 n=num_return 生成多个输出）
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=num_return,                     # 关键：生成多个候选
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        # 纯文本输入，不需要 multi_modal_data
        inputs = [{"prompt": p} for p in prompts]

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        # 解析结果
        results = []
        for output in outputs:
            if output and output.outputs:
                # output.outputs 是一个列表，长度等于 n
                candidates = [o.text.strip() for o in output.outputs]
                # 简单过滤空文本
                candidates = [c for c in candidates if c]
                # 如果数量不足 num_return，用空字符串补足？此处直接保留实际生成的
                results.append(candidates)
            else:
                results.append([])  # 生成失败，返回空列表

        return results


# ------------------------------
# 数据集扫描函数（基于 meta.json 精确选取）
# ------------------------------
def get_all_images(dataset_root: str) -> List[Tuple[str, bool, str]]:
    """
    从 meta.json 中精确选取 141 张图像：
    - 每个类别 1 张正常图像（good）
    - 每个类别每种异常类型各 1 张图像（必须有 mask）

    Returns:
        List of (img_abs_path, is_normal, mask_abs_path)
        正常样本的 mask_abs_path 为 ""
    """
    meta_path = os.path.join(dataset_root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images = []
    # meta.json 中 train 为空，所有数据在 test 中
    categories = sorted(meta["test"].keys())

    for category in categories:
        entries = meta["test"][category]
        if not entries:
            continue

        # 1. 选 1 张正常图像（good）
        good_entries = [e for e in entries if e["specie_name"] == "good"]
        if good_entries:
            chosen = good_entries[0]
            img_path = os.path.join(dataset_root, chosen["img_path"])
            images.append((img_path, True, ""))

        # 2. 每种异常类型选 1 张（必须有 mask）
        defect_types = sorted(set(
            e["specie_name"] for e in entries if e["specie_name"] != "good"
        ))
        for defect_type in defect_types:
            candidates = [
                e for e in entries
                if e["specie_name"] == defect_type and e.get("mask_path", "")
            ]
            if candidates:
                chosen = candidates[0]
                img_path = os.path.join(dataset_root, chosen["img_path"])
                mask_path = os.path.join(dataset_root, chosen["mask_path"])
                images.append((img_path, False, mask_path))
            else:
                print(f"警告: {category}/{defect_type} 无带 mask 的样本，跳过")

    normal_count = sum(1 for _, is_n, _ in images if is_n)
    abnormal_count = len(images) - normal_count
    print(f"从 meta.json 选取 {len(images)} 张图像（正常: {normal_count}, 异常: {abnormal_count}）")
    return images


# ------------------------------
# 生成所有图片的描述（带缓存）
# ------------------------------
def generate_all_descriptions(
    descriptor: QwenVLDescriptor,
    images: List[Tuple[str, bool, str]],
    temperatures: List[float],
    cache_file: str,
    batch_size: int,
) -> Dict[str, Dict]:
    """生成多温度描述，并缓存。images 中每个元素为 (img_path, is_normal, mask_path)。"""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"从缓存加载 {len(cache)} 张图片的描述")
    else:
        cache = {}

    to_process = []
    for img_path, is_normal, mask_path in images:
        if img_path not in cache:
            to_process.append((img_path, is_normal, mask_path))

    if not to_process:
        print("所有图片均已缓存，无需生成")
        return cache

    print(f"需要生成 {len(to_process)} 张图片的描述...")

    for i in tqdm(range(0, len(to_process), batch_size), desc="生成描述批次"):
        batch = to_process[i:i+batch_size]
        batch_paths = [p for p, _, _ in batch]
        batch_is_normal = [n for _, n, _ in batch]
        batch_masks = [m for _, _, m in batch]

        results_by_temp = descriptor.generate_multi_temp(
            batch_paths, batch_is_normal, temperatures,
            mask_paths=batch_masks
        )

        for temp, results in results_by_temp.items():
            for res in results:
                img_path = res["image_path"]
                if img_path not in cache:
                    cache[img_path] = {
                        "label": res["is_normal"],
                        "descriptions": []
                    }
                if res["parsed"]["parsing_success"]:
                    desc_text = f"{res['parsed']['description']} {res['parsed']['think']}".strip()
                    cache[img_path]["descriptions"].append({
                        "text": desc_text,
                        "temperature": temp,
                        "raw_response": res["raw_response"]
                    })
                else:
                    print(f"解析失败 {img_path} (temp={temp}): {res['parsed'].get('error')}")

        if (i // batch_size) % 5 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"描述缓存已保存至 {cache_file}")
    return cache


# ------------------------------
# 构造句子对（包括大模型改写生成的样本）
# ------------------------------
def construct_pairs(
    desc_cache: Dict[str, Dict],
    descriptor: Optional[QwenVLDescriptor] = None,   # 用于改写
    num_random_negatives: Optional[int] = None,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    num_rewrites: int = 1,                # 每个温度生成多少个改写（此处每个温度只生成1个）
    rewrite_temperatures: Optional[List[float]] = None,  # 修改：新增改写温度列表
) -> List[Dict]:
    """
    构造句子对，包括：
    - 正例：原描述 + 相似改写（每个温度一个）
    - 负例：不同标签随机配对
    - 负例：关键词替换（遍历替换）
    - 负例：原描述 + 相反改写（每个温度一个）
    """
    pairs = []

    # 收集所有描述，按标签分组（用于随机负例）
    normal_texts = []   # 元素为 (img_path, text)
    abnormal_texts = []
    # 同时收集所有描述及其原始信息（用于改写）
    all_descriptions = []  # 元素为 (img_path, text, label)

    for img_path, data in desc_cache.items():
        label = data["label"]  # True 正常，False 异常
        for desc in data["descriptions"]:
            text = desc["text"]
            all_descriptions.append((img_path, text, label))
            if label:
                normal_texts.append((img_path, text))
            else:
                abnormal_texts.append((img_path, text))

    print(f"正常描述数量: {len(normal_texts)}, 异常描述数量: {len(abnormal_texts)}")

    # 随机负例：不同标签的描述随机配对
    if num_random_negatives is None:
        num_random_negatives = len(pairs) // 2 if pairs else 1000
    for _ in range(num_random_negatives):
        if normal_texts and abnormal_texts:
            _, t1 = random.choice(normal_texts)
            _, t2 = random.choice(abnormal_texts)
            pairs.append({
                "text1": t1,
                "text2": t2,
                "label": 0,
                "type": "random_negative"
            })

    # 关键词替换负例（遍历替换）
    if keyword_map:
        for img_path, data in desc_cache.items():
            for desc in data["descriptions"]:
                text = desc["text"]
                replaced_texts = traverse_replace(text, keyword_map)
                for replaced_text in replaced_texts:
                    if replaced_text != text:   # 确保不同
                        pairs.append({
                            "text1": text,
                            "text2": replaced_text,
                            "label": 0,
                            "type": "keyword_negative"
                        })

    # ---------- 修改：大模型改写生成样本（遍历温度列表） ----------
    if descriptor is not None and num_rewrites > 0:
        # 设置默认温度列表
        if rewrite_temperatures is None:
            rewrite_temperatures = DEFAULT_REWRITE_TEMPERATURES
        texts = [t for _, t, _ in all_descriptions]
        print(f"开始生成大模型改写样本，温度列表: {rewrite_temperatures}，每个温度生成相似/相反各 {num_rewrites} 个...")

        # 遍历每个温度
        for temp in rewrite_temperatures:
            # 生成相似改写
            similar_rewrites = descriptor.rewrite_texts(
                texts,
                rewrite_type="similar",
                temperature=temp,
                num_return=num_rewrites,
                max_tokens=DEFAULT_MAX_REWRITE_TOKENS
            )
            # 配对：原始 + 相似改写 → 正例
            for (img_path, orig_text, label), rewrites in zip(all_descriptions, similar_rewrites):
                for r in rewrites:
                    if r and r != orig_text:   # 避免空或相同
                        pairs.append({
                            "text1": orig_text,
                            "text2": r,
                            "label": 1,
                            "type": f"similar_rewrite_temp_{temp}"
                        })

            # 生成相反改写
            opposite_rewrites = descriptor.rewrite_texts(
                texts,
                rewrite_type="opposite",
                temperature=temp,
                num_return=num_rewrites,
                max_tokens=DEFAULT_MAX_REWRITE_TOKENS
            )
            # 配对：原始 + 相反改写 → 负例
            for (img_path, orig_text, label), rewrites in zip(all_descriptions, opposite_rewrites):
                for r in rewrites:
                    if r and r != orig_text:
                        pairs.append({
                            "text1": orig_text,
                            "text2": r,
                            "label": 0,
                            "type": f"opposite_rewrite_temp_{temp}"
                        })

    # 打乱顺序（可选）
    # random.shuffle(pairs)
    return pairs


# ------------------------------
# 主函数
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="为 real-iad 数据集生成句子对（基于 Qwen3-VL 描述 + 大模型改写）")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Qwen3-VL 模型路径")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="real-iad 数据集根目录，包含 OK 和 NG 子目录")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="输出句子对 JSONL 文件")
    parser.add_argument("--desc-cache", type=str, default=DEFAULT_DESC_CACHE,
                        help="描述缓存文件")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="批处理大小")
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE,
                        help="GPU 并行数量")
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES,
                        help="用于生成多样性的 temperature 列表（注：此参数已强制覆盖为[0.0]）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子")
    parser.add_argument("--num-random-negatives", type=int, default=None,
                        help="随机负例数量（默认自动计算）")

    # 新增参数
    parser.add_argument("--num-rewrites", type=int, default=DEFAULT_NUM_REWRITES,
                        help="每条描述每个温度生成的相似/相反改写数量（默认1）")
    parser.add_argument("--rewrite-temperatures", type=float, nargs="+", default=DEFAULT_REWRITE_TEMPERATURES,
                        help="改写温度列表（默认0.1,0.2,...,1.0）")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="禁用大模型改写（只使用原有方式）")

    args = parser.parse_args()
    random.seed(args.seed)

    # 1. 从 meta.json 选取图片
    print("正在从 meta.json 选取图片...")
    images = get_all_images(args.dataset_root)
    print(f"共选取 {len(images)} 张图片 (正常: {sum(1 for _, n, _ in images if n)}, 异常: {sum(1 for _, n, _ in images if not n)})")

    # 2. 初始化描述器
    descriptor = QwenVLDescriptor(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size
    )

    # 3. 生成描述
    desc_cache = generate_all_descriptions(
        descriptor=descriptor,
        images=images,
        temperatures=[0.0],  # 固定使用温度0.0，忽略 args.temperatures
        cache_file=args.desc_cache,
        batch_size=args.batch_size
    )

    # 4. 构造句子对（如果不禁用改写，则传入 descriptor 用于改写）
    if args.no_rewrite:
        pairs = construct_pairs(
            desc_cache,
            descriptor=None,
            num_random_negatives=args.num_random_negatives,
            keyword_map=KEYWORD_MAP
        )
    else:
        pairs = construct_pairs(
            desc_cache,
            descriptor=descriptor,
            num_random_negatives=args.num_random_negatives,
            keyword_map=KEYWORD_MAP,
            num_rewrites=args.num_rewrites,
            rewrite_temperatures=args.rewrite_temperatures  # 使用温度列表
        )
    print(f"共构造 {len(pairs)} 个句子对")

    # 5. 保存为 JSONL
    with open(args.output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"句子对已保存至 {args.output_file}")

    # 统计各类别数量
    type_count = {}
    for p in pairs:
        t = p.get("type", "unknown")
        type_count[t] = type_count.get(t, 0) + 1
    print("各类别样本数量：", type_count)


if __name__ == "__main__":
    main()