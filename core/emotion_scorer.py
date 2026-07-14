"""表情识别与笑容打分（基于 FerPlus ONNX 模型）。

FerPlus 输出 8 类 softmax 概率，索引对应：
0=Neutral, 1=Happy, 2=Surprise, 3=Sad, 4=Anger,
5=Disgust, 6=Fear, 7=Contempt

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

    Args:
        probs: 长度为 8 的概率向量。
        surprise_weight: surprise 概率的权重（0-1）。

    Returns:
        0-100 的整数分数。
    """
    happy = probs[1]
    surprise = probs[2]
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
        # 预处理：转灰度 → resize → 标准化 → NCHW float32
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.INPUT_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        # FerPlus 训练时用 mean=0，std=1；不做 mean subtraction（按官方示例）
        input_tensor = normalized.reshape(1, 1, 64, 64)

        # 推理
        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits = outputs[0][0]

        # softmax 转为概率
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        probs_list = probs.tolist()

        score = compute_smile_score(probs_list, self._surprise_weight)
        return score, probs_list
