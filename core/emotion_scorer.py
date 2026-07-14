"""表情识别与笑容打分（基于 FerPlus ONNX 模型）。

FerPlus 输出 8 类 softmax 概率。**实测的索引顺序**（通过对比实验验证）：
0=Happiness (笑), 1=Neutral, 2=Surprise, 3=Sadness, 4=Anger,
5=Disgust, 6=Fear, 7=Contempt

注：ONNX 模型文档说索引 0=neutral, 1=happy，但实测输出顺序是反的（索引 0 才是 happy）。
    通过笑脸图 vs 不笑图的对照实验确认（happy 图在索引 0 概率 0.81，不笑图在索引 1 概率 0.96）。

分数公式：score = (happy + surprise_weight * surprise) * 100
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

    索引对应（实测）：
        0 = happiness
        1 = neutral
        2 = surprise
        3 = sadness
        ...

    Args:
        probs: 长度为 8 的概率向量。
        surprise_weight: surprise 概率的权重（0-1）。

    Returns:
        0-100 的整数分数。
    """
    happy = probs[0]      # 实测：索引 0 才是 happiness
    surprise = probs[2]   # 索引 2 是 surprise
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
