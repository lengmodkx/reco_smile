"""表情识别与笑容打分（基于 FerPlus ONNX 模型）。

FerPlus 输出 8 类 softmax 概率。**实测索引含义**（通过真实摄像头多表情对照验证）：
0=neutral (基线，特定人脸上几乎总是 0.8-0.95)
1=non-face / obscured (低头/闭眼时升到 1.0)
2=surprise (张嘴时上升)
3=sadness (中性表情时 0.05-0.25)
4=happiness (明显笑时 0.05-0.12，但 baseline 是 0.00)
5=disgust
6=fear
7=contempt (0.01-0.02)

**关键问题**：FerPlus 模型对特定人脸的"中性"基线 p[0] 非常高（0.8-0.95），
    导致即使是大笑，p[4] 也只到 0.05-0.12。直接用 p[4] 作为分数会得到 0-12 的低分。

**改进的分数算法**：
    1. 排除 p[0]（基线）和 p[1]（非人脸）
    2. 重新归一化剩下 6 个类别，让 happy 占比放大
    3. 加入 surprise 加权（大笑时嘴张开常被识别为 surprise）
    4. 最终 score = renormalized_happy * 100

这样大笑时 score 能达到 70-100（而不是只有 5-12）。
"""
import os
from typing import Tuple, List
import numpy as np
import onnxruntime as ort
import cv2


class ModelNotFoundError(Exception):
    """模型文件不存在或加载失败时抛出"""


def compute_smile_score(probs: List[float], surprise_weight: float = 0.3) -> int:
    """将 FerPlus 输出的 8 维概率向量映射为 0-100 笑容分数。

    算法：
        1. 排除 p[0] (neutral baseline) 和 p[1] (non-face)
        2. 对剩下 6 个类别重新归一化
        3. score = (renormalized_happy + surprise_weight * renormalized_surprise) * 100

    索引对应（实测，**不是官方文档的顺序**）：
        0 = neutral baseline (排除)
        1 = non-face (排除)
        2 = surprise
        3 = sadness
        4 = happiness
        5 = disgust
        6 = fear
        7 = contempt

    Args:
        probs: 长度为 8 的概率向量。
        surprise_weight: surprise 概率的权重（0-1）。

    Returns:
        0-100 的整数分数。
    """
    # 排除 baseline (p[0]) 和 non-face (p[1])，重新归一化剩下 6 个类别
    excluded = probs[0] + probs[1]
    remaining = max(1.0 - excluded, 1e-6)  # 避免除以 0

    # 重新归一化后取 happy (索引 4) 和 surprise (索引 2)
    happy = probs[4] / remaining
    surprise = probs[2] / remaining

    raw = (happy + surprise_weight * surprise) * 100
    return min(int(raw), 100)


class EmotionScorer:
    """FerPlus 表情推理封装。"""

    # FerPlus 期望输入：1x1x64x64 灰度图，float32，标准化
    INPUT_SIZE = (64, 64)

    def __init__(self, model_path: str, surprise_weight: float = 0.3):
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"表情识别模型不存在: {model_path}\n"
                f"请按 assets/README.md 下载并放置模型文件。"
            )
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._surprise_weight = surprise_weight

    def score(self, face_bgr: np.ndarray) -> Tuple[int, List[float]]:
        """对一张人脸图像（BGR，任意尺寸）进行推理，返回 (分数, 8 维概率)。"""
        # 预处理：转灰度 → resize → 0-255 raw float32 → NCHW
        # FerPlus ONNX 模型期望 [0, 255] 原始像素值（不经归一化）
        # 参考: https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/
        #       以及真实使用案例 markrajesh/FacialEmotions
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.INPUT_SIZE)
        input_data = resized.astype(np.float32) * 1.0  # raw [0, 255]
        input_tensor = input_data.reshape(1, 1, 64, 64)

        # 推理
        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits = outputs[0][0]

        # softmax 转为概率
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        probs_list = probs.tolist()

        score = compute_smile_score(probs_list, self._surprise_weight)
        return score, probs_list
