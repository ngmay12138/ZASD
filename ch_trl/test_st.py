import re
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


class SemanticJudge:
    """
    语义判断器：根据标准答案和预测答案，使用Sentence Transformer计算语义相似度。
    输出1表示语义相同，0表示不同。
    """

    def __init__(
        self,
        model_path: str = "all-MiniLM-L6-v2",
        threshold: float = 0.8,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化Sentence Transformer模型。

        Args:
            model_path: 预训练模型名称或本地路径（兼容SentenceTransformer支持的模型）
            threshold: 判断语义相同的相似度阈值，默认0.8
            device: 运行设备（'cuda'、'cpu'等），默认自动选择
            **kwargs: 其他参数（预留）
        """
        self.model_path = model_path
        self.threshold = threshold
        self.device = device
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """加载Sentence Transformer模型"""
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            **kwargs
        )
        print(f"Sentence Transformer模型加载完成，模型路径: {self.model_path}")

    def judge(self, gt: str, pred: str) -> int:
        """
        对单对文本进行语义判断。

        Args:
            gt: 标准答案文本
            pred: 预测答案文本

        Returns:
            1（语义相同）或0（语义不同）
        """
        # 编码句子对，返回嵌入向量
        # 设置truncation=True自动截断超出最大长度的部分
        embeddings = self.model.encode(
            [gt, pred],
            convert_to_tensor=True,
        )
        # 计算余弦相似度
        cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        print(f"余弦相似度：{cos_sim}") 
        # 根据阈值返回结果
        return 1 if cos_sim >= self.threshold else 0

########################################改写相似度0.95，错误判断0.99######################
# 使用示例
if __name__ == "__main__":
    judge = SemanticJudge(model_path="all-MiniLM-L6-v2", threshold=0.8)
    gt_text = "Upon examining the image, I observe a uniform background texture consistent with a natural material like concrete or stone, featuring small speckles and variations in color. However, there is a distinct, relatively large, uniformly dark circular spot at the center, which stands out significantly from the surrounding texture. This central spot appears to be a localized discoloration or depression, possibly indicating a stain, pit, or early-stage corrosion. Such an anomaly is not typical of a uniform, natural surface and represents a deviation from the expected homogeneity. In industrial anomaly detection, any unexpected localized defect—especially one that disrupts the surface’s uniformity— is considered a potential failure or damage. Therefore, the presence of this central dark spot classifies the object as abnormal."
    # pred_text = "Upon inspecting the image, the background displays a consistent texture resembling a natural surface such as concrete or stone, with subtle color variations and small speckles. At the center, however, a notably large, uniformly dark circular region draws attention due to its contrast with the surrounding area. This dark spot appears to be a localized irregularity—possibly a stain, pit, or early corrosion—suggesting a physical or chemical change. In the context of industrial anomaly detection, such a distinct deviation from the surface's expected uniformity is regarded as a potential defect. Consequently, the presence of this central anomaly leads to the classification of the object as abnormal."
    pred_text = "Upon examining the image, I observe a uniform background texture consistent with a natural material like concrete or stone, featuring small speckles and variations in color. However, there is a distinct, relatively large, uniformly dark circular spot at the center, which stands out significantly from the surrounding texture. This central spot appears to be a localized discoloration or depression, possibly indicating a stain, pit, or early-stage corrosion. Such an anomaly is not typical of a uniform, natural surface and represents a deviation from the expected homogeneity. In industrial anomaly detection, any unexpected localized defect—especially one that disrupts the surface’s uniformity— is considered a potential failure or damage. Therefore, the presence of this central dark spot classifies the object as normal."
    gt_text = "这个苹果是红色的."
    pred_text = "这个苹果是红色的吗"
    # pred_text = "这个苹果的颜色是红色"
    
    result = judge.judge(gt_text, pred_text)
    print(f"判断结果：{result}")  # 根据相似度输出1或0