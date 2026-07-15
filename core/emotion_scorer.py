"""笑容打分（基于几何特征，不再使用 FerPlus 模型）。

原理：使用 YuNet 人脸检测器提供的 5 个关键点：
- 右眼中心 (re_x, re_y)
- 左眼中心 (le_x, le_y)
- 鼻尖 (n_x, n_y)
- 嘴部右角 (mr_x, mr_y)
- 嘴部左角 (ml_x, ml_y)

v5 算法：**基于'嘴部张开度/眼距'作为主特征**。

用户实测发现（这是最可靠的特征）：
- 闭嘴/不笑: mouth_h/eye_d ≈ 0.014-0.027 （嘴部高度小）
- 露齿大笑:  mouth_h/eye_d ≈ 0.06-0.12   （嘴部张开，露齿）

差异 3-4 倍！比"嘴部左右角位置变化"或"嘴部上下移动"更可靠。

同时辅以：
- mouth_w/eye_d (嘴宽)
- eye_dist (眯眼 = 眼距变小)

v4 失败原因：v4 用嘴部 Y 位置变化作为信号，但用户实测中嘴部 Y 几乎不变，
            只在张嘴大笑时才有微小变化。基线对比的方法不可靠。
"""
import os
from typing import Tuple, List, Optional
import numpy as np
import cv2


class ModelNotFoundError(Exception):
    """占位异常（保持 API 兼容，但不再需要模型）"""


def compute_smile_score(
    mouth_width_ratio: float,
    eye_to_mouth_y_ratio: float,
    mouth_up_score: float = 0.0,
    mouth_width_score: float = 0.0,
) -> int:
    """根据多个几何特征计算 0-100 笑容分数（v3）。

    v3 算法：直接接受已经算好的 0-100 子分数，求平均。
    - mouth_up_score: 嘴部上移分数（eye_to_mouth_y_ratio 越小越高）
    - mouth_width_score: 嘴宽分数（mouth_width_ratio 越大越高）

    Args:
        mouth_width_ratio: 嘴宽 / 眼距（用于 info/debug）。
        eye_to_mouth_y_ratio: 嘴部 Y 距眼睛 Y（用于 info/debug）。
        mouth_up_score: 嘴部上移 0-100 分数。
        mouth_width_score: 嘴宽 0-100 分数。

    Returns:
        0-100 的整数分数。
    """
    # 综合两个子分数
    score = (mouth_up_score + mouth_width_score) / 2
    return min(int(score), 100)


class EmotionScorer:
    """笑容打分器（v4: 相对基线对比）。

    不再使用 FerPlus ONNX 模型。v4 算法：
    1. 启动后采集一帧"基线"（用户在不笑状态下的嘴部几何特征）
    2. 之后每帧跟基线对比，差异越大分数越高

    这样能自动适配每个人不同的脸部基线特征，绕开 YuNet 5 关键点的绝对值不准问题。
    """

    def __init__(self, ferplus_path: str = "", surprise_weight: float = 0.5):
        self._surprise_weight = surprise_weight
        self._baseline = None  # 基线特征 (mouth_width_ratio, eye_to_mouth_y_ratio)
        self._baseline_frames = 0  # 累计基线帧数

    def set_baseline(self, mouth_width_ratio: float, eye_to_mouth_y_ratio: float):
        """手动设置基线（用户初始不笑时的几何特征）。"""
        self._baseline = (mouth_width_ratio, eye_to_mouth_y_ratio)
        self._baseline_frames = 30  # 立即认为基线稳定

    def _update_baseline(self, mouth_width_ratio: float, eye_to_mouth_y_ratio: float):
        """前 30 帧自动建立基线（取中位数）。"""
        if not hasattr(self, "_samples"):
            self._samples = []
        self._samples.append((mouth_width_ratio, eye_to_mouth_y_ratio))
        self._baseline_frames += 1

        if self._baseline_frames == 30:
            # 用最后 30 帧的中位数作为基线
            samples = self._samples[-30:]
            mw = sorted([s[0] for s in samples])
            em = sorted([s[1] for s in samples])
            baseline_mw = mw[len(mw) // 2]
            baseline_em = em[len(em) // 2]
            self._baseline = (baseline_mw, baseline_em)

    def score_from_landmarks(
        self,
        re_x: float, re_y: float,
        le_x: float, le_y: float,
        mr_x: float, mr_y: float,
        ml_x: float, ml_y: float,
        face_w: float = 0,
        face_h: float = 0,
        nose_y: float = 0,
    ) -> Tuple[int, dict]:
        """从 5 个关键点计算笑容分数（v7 - 综合多个独立信号）。

        v6 失败经验: 单个几何特征在 YuNet 5 关键点下反应微弱。
        v7 算法: 综合 4 个独立信号，取**最大**而不是平均——这样
        即便只有一个信号激活（用户某种类型的笑）也能给出高分。

        4 个独立信号：
        A. 嘴角外推 (mouth_extent_ratio) - 露齿笑
        B. 嘴部 Y 差 (mouth_h_ratio) - 张嘴大笑
        C. 鼻尖到嘴部距离 - 嘴部位置变化
        D. 嘴部宽度变化 (mouth_w_ratio) - 辅助

        score = max(score_a, score_b, score_c, score_d)
        """
        if face_w <= 0:
            return 0, {"error": "face_w invalid"}

        eye_dist = max(abs(re_x - le_x), 1)

        # === A. 嘴角外推 ===
        mouth_extent_x = max(mr_x, ml_x) - min(mr_x, ml_x)
        mouth_extent_ratio = mouth_extent_x / face_w
        score_a = max(0, min(100, (mouth_extent_ratio - 0.40) / 0.10 * 100))

        # === B. 嘴部 Y 差（张嘴）===
        mouth_h = abs(mr_y - ml_y)
        mouth_h_ratio = mouth_h / eye_dist
        score_b = max(0, min(100, (mouth_h_ratio - 0.02) / 0.10 * 100))

        # === C. 鼻尖到嘴部距离（嘴部位置变化）===
        if nose_y > 0 and face_h > 0:
            mouth_center_y = (mr_y + ml_y) / 2
            nose_to_mouth = abs(mouth_center_y - nose_y)
            n2m_ratio = nose_to_mouth / face_h
            # 不笑时 n2m ≈ 0.13-0.15, 微笑时 ≈ 0.16, 大笑时 ≈ 0.18+
            # 阈值 0.16 (轻微提高)
            score_c = max(0, min(100, (n2m_ratio - 0.16) / 0.04 * 100))
        else:
            score_c = 0

        # === D. 嘴宽变化 ===
        mouth_width = np.sqrt((mr_x - ml_x) ** 2 + (mr_y - ml_y) ** 2)
        mouth_width_ratio = mouth_width / eye_dist
        # 不笑: 0.78, 大笑: 0.92
        score_d = max(0, min(100, (mouth_width_ratio - 0.78) / 0.14 * 100))

        # 取最大值（更激进）
        score = int(max(score_a, score_b, score_c, score_d))

        return score, {
            "mouth_extent_ratio": mouth_extent_ratio,
            "mouth_h_ratio": mouth_h_ratio,
            "n2m_ratio": n2m_ratio if nose_y > 0 else 0,
            "mouth_width_ratio": mouth_width_ratio,
            "score_a": score_a,
            "score_b": score_b,
            "score_c": score_c,
            "score_d": score_d,
        }

    def score(self, face_bgr: np.ndarray) -> Tuple[int, List[float]]:
        """兼容旧 API。"""
        return 0, [0.0] * 8
