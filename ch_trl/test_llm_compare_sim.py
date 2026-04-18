import re
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class SemanticJudge:
    """
    语义判断器：根据标准答案和预测答案，使用LLM判断两者语义是否相同。
    输出1表示相同，0表示不同。
    """

    def __init__(self, model_path: str = "qwen3_4b", tensor_parallel_size: int = 1, **kwargs):
        """
        初始化LLM考官模型。

        Args:
            model_path: 模型路径或名称
            tensor_parallel_size: 张量并行大小
            **kwargs: 其他参数（预留）
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._init_llm()

    def _init_llm(self):
        """加载tokenizer并初始化vLLM引擎"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",  # 可根据实际环境调整
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.2,
            max_model_len=2048,
        )
        # 设置生成参数：温度0保证确定性输出，只需生成少量tokens
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        print(f"LLM考官初始化完成，模型路径: {self.model_path}")

    def _build_prompt(self, gt: str, pred: str) -> str:
        """构建用于判断的prompt"""
        template = (
            "You are an industrial anomaly detection expert. Please determine, "
            "based on the provided 'standard answer,' whether the 'predicted answer' "
            "is semantically the same as the 'standard answer.' If they are the same, "
            "output 1; otherwise, output 0.\n"
            "Standard answer：{gt}\n"
            "Predicted answer：{pred}\n\n"
            "Output："

        )
        return template.format(gt=gt, pred=pred)

    def _llm_evaluate(self, prompts: List[str]) -> List[int]:
        """批量执行LLM推理，解析输出中的0/1"""
        if not prompts:
            return []
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores = []
        for output in outputs:
            if output and output.outputs:
                response = output.outputs[0].text.strip()
                # 提取第一个出现的数字0或1
                match = re.search(r'[01]', response)
                if match:
                    score = int(match.group())
                else:
                    score = 0  # 默认视为不同
            else:
                score = 0
            scores.append(score)
        return scores

    def judge(self, gt: str, pred: str) -> int:
        """
        对单对文本进行语义判断。

        Args:
            gt: 标准答案文本
            pred: 预测答案文本

        Returns:
            1（语义相同）或0（语义不同）
        """
        prompt = self._build_prompt(gt, pred)
        scores = self._llm_evaluate([prompt])
        return scores[0] if scores else 0

########################################改写相似为1，错误判断为0###############################
if __name__ == "__main__":
    judge = SemanticJudge(model_path="Qwen3-VL-2B")
    gt_text = "Upon examining the image, I observe a uniform background texture consistent with a natural material like concrete or stone, featuring small speckles and variations in color. However, there is a distinct, relatively large, uniformly dark circular spot at the center, which stands out significantly from the surrounding texture. This central spot appears to be a localized discoloration or depression, possibly indicating a stain, pit, or early-stage corrosion. Such an anomaly is not typical of a uniform, natural surface and represents a deviation from the expected homogeneity. In industrial anomaly detection, any unexpected localized defect—especially one that disrupts the surface’s uniformity— is considered a potential failure or damage. Therefore, the presence of this central dark spot classifies the object as abnormal."
    pred_text = "Upon inspecting the image, the background displays a consistent texture resembling a natural surface such as concrete or stone, with subtle color variations and small speckles. At the center, however, a notably large, uniformly dark circular region draws attention due to its contrast with the surrounding area. This dark spot appears to be a localized irregularity—possibly a stain, pit, or early corrosion—suggesting a physical or chemical change. In the context of industrial anomaly detection, such a distinct deviation from the surface's expected uniformity is regarded as a potential defect. Consequently, the presence of this central anomaly leads to the classification of the object as abnormal."
    # pred_text = "Upon examining the image, I observe a uniform background texture consistent with a natural material like concrete or stone, featuring small speckles and variations in color. However, there is a distinct, relatively large, uniformly dark circular spot at the center, which stands out significantly from the surrounding texture. This central spot appears to be a localized discoloration or depression, possibly indicating a stain, pit, or early-stage corrosion. Such an anomaly is not typical of a uniform, natural surface and represents a deviation from the expected homogeneity. In industrial anomaly detection, any unexpected localized defect—especially one that disrupts the surface’s uniformity— is considered a potential failure or damage. Therefore, the presence of this central dark spot classifies the object as normal."
    gt_text = "这个苹果是红色的."
    # pred_text = "这个苹果是红色的吗"
    pred_text = "这个苹果的颜色是红色"
    
    
    result = judge.judge(gt_text, pred_text)
    print(f"判断结果：{result}")  # 预期输出1或0