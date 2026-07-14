"""笑容打分（基于几何特征，不再使用 FerPlus 模型）。

原理：使用 YuNet 人脸检测器提供的 5 个关键点：
- 右眼中心 (re_x, re_y)
- 左眼中心 (le_x, le_y)
- 鼻尖 (n_x, n_y)
- 嘴部右角 (mr_x, mr_y)
- 嘴部左角 (ml_x, ml_y)

几何特征：
- 眼距 = 两眼距离（用于归一化）
- 嘴宽 = 两嘴角距离
- 嘴部 Y 相对眼睛的偏移 = mouth_avg_y - eye_avg_y
- 嘴部 X 中心相对鼻尖 = mouth_center_x - nose_x

分数计算思路：
- 嘴角上扬：嘴宽增加（smile: mouth_width ratio 增加）
- 露齿笑：嘴部 Y 会上移（露齿时嘴角位置略低但嘴部中点也受影响）
- 眯眼：双眼 Y 会上移（笑眯眼）

由于 MVP 阶段只关心"笑不笑"，我们用嘴宽/眼距作为主指标：
- 嘴宽/眼距 < 0.7: 闭嘴/嘴角下压 → 0-20
- 嘴宽/眼距 0.7-0.9: 中性 → 20-50
- 嘴宽/眼距 0.9-1.1: 微笑 → 50-80
- 嘴宽/眼距 > 1.1: 大笑 → 80-100
"""
import os
from typing import Tuple, List, Optional
import numpy as np
import cv2


class ModelNotFoundError(Exception):
    """占位异常（保持 API 兼容，但不再需要模型）"""


def compute_smile_score(
    mouth_width_ratio: float,
    eye_to_mouth_y_ratio: float = 0.0,
    surprise_weight: float = 0.3,
) -> int:
    """根据嘴宽/眼距计算 0-100 笑容分数。

    算法（基于实测基线 ~0.80~0.95，嘴部张开会进一步增加分数）：
        - 0.75: 嘴角下压 → 0 分
        - 0.85: 中性 → 20 分
        - 0.95: 明显微笑 → 60 分
        - 1.00+: 大笑 → 80 分
        - 1.05+: 露齿大笑 → 100 分
        线性映射：score = (ratio - 0.75) / 0.30 * 100

    Args:
        mouth_width_ratio: 嘴宽 / 眼距。
        eye_to_mouth_y_ratio: (mouth_avg_y - eye_avg_y) / eye_dist，
            张嘴大笑时嘴部 Y 会上移（值变小），用来加 bonus。
        surprise_weight: 张嘴加权的强度（0-1）。

    Returns:
        0-100 的整数分数。
    """
    # 主信号：嘴宽比，0.75 → 0 分，1.05 → 100 分
    base_score = (mouth_width_ratio - 0.75) / 0.30 * 100
    base_score = max(0, min(100, base_score))

    # 张嘴 bonus：eye_to_mouth_y_ratio 变小说明嘴部上移（张嘴）
    # 典型值: 闭嘴时 ~0.85, 张嘴时 ~0.65
    if eye_to_mouth_y_ratio > 0:
        y_bonus = max(0, (0.85 - eye_to_mouth_y_ratio) / 0.20) * surprise_weight * 20
        base_score = base_score + y_bonus

    return min(int(base_score), 100)


class EmotionScorer:
    """笑容打分器（几何特征版）。

    不再使用 FerPlus ONNX 模型。改用 YuNet 检测器返回的 5 个关键点
    计算嘴宽/眼距作为主指标。

    为保持 API 兼容，仍然接收 ferplus_path 参数但不使用。
    """

    def __init__(self, ferplus_path: str = "", surprise_weight: float = 0.3):
        # ferplus_path 参数保留但不使用（保持 API 兼容）
        self._surprise_weight = surprise_weight

    def score_from_landmarks(
        self,
        re_x: float, re_y: float,
        le_x: float, le_y: float,
        mr_x: float, mr_y: float,
        ml_x: float, ml_y: float,
    ) -> Tuple[int, dict]:
        """从 5 个关键点计算笑容分数。

        Returns:
            (score, info_dict)
            info_dict 包含 mouth_width_ratio, eye_to_mouth_y_ratio 等调试信息
        """
        eye_dist = np.sqrt((re_x - le_x) ** 2 + (re_y - le_y) ** 2)
        if eye_dist < 1:
            # 关键点异常
            return 0, {"mouth_width_ratio": 0, "eye_to_mouth_y_ratio": 0, "error": "eye_dist too small"}

        mouth_width = np.sqrt((mr_x - ml_x) ** 2 + (mr_y - ml_y) ** 2)
        mouth_width_ratio = mouth_width / eye_dist

        eye_avg_y = (re_y + le_y) / 2
        mouth_avg_y = (mr_y + ml_y) / 2
        eye_to_mouth_y_ratio = (mouth_avg_y - eye_avg_y) / eye_dist

        score = compute_smile_score(
            mouth_width_ratio,
            eye_to_mouth_y_ratio,
            self._surprise_weight,
        )

        return score, {
            "mouth_width_ratio": mouth_width_ratio,
            "eye_to_mouth_y_ratio": eye_to_mouth_y_ratio,
            "mouth_width": mouth_width,
            "eye_dist": eye_dist,
        }

    def score(self, face_bgr: np.ndarray) -> Tuple[int, List[float]]:
        """兼容旧 API：接受人脸图像，但因没有 5 点关键点，返回 0。

        实际使用时应该用 score_from_landmarks。
        """
        # 不再支持从图像直接打分（几何方案需要 landmark）
        return 0, [0.0] * 8
