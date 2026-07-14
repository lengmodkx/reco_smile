"""笑容打分（基于几何特征，不再使用 FerPlus 模型）。

原理：使用 YuNet 人脸检测器提供的 5 个关键点：
- 右眼中心 (re_x, re_y)
- 左眼中心 (le_x, le_y)
- 鼻尖 (n_x, n_y)
- 嘴部右角 (mr_x, mr_y)
- 嘴部左角 (ml_x, ml_y)

v3 算法：综合多个几何特征。
原因：v2 用嘴宽/眼距作为主信号，但实测发现 YuNet 的"嘴部左右角"标注
对露齿大笑时不稳定（mouth_w/eye_d 跟不笑几乎一样）。
改用以下三个**大笑时变化显著**的几何特征：

1. **eye_to_mouth_y_ratio** (主信号):
   嘴部 Y 距眼睛 Y 的距离（归一化）。
   - 不笑: ~1.15-1.18
   - 大笑（嘴部上移）: ~1.05-1.10
   - 越小 = 越笑

2. **eye_dist / face_height_proxy** (眯眼信号):
   双眼距离相对脸部高度的归一化。
   - 眯眼（笑眯眼）时眼距变小
   - 越笑 = 比值越小

3. **mouth_width_ratio** (辅助信号):
   嘴宽/眼距。
   - 笑时嘴会变宽

分数公式：
    score = mouth_up_score + eye_squeeze_score + mouth_width_score
    其中各项用线性映射到 0-100 后求平均

这样大笑时（嘴部上移+眯眼+嘴宽）三项都激活，分数可达 80+。
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
    ) -> Tuple[int, dict]:
        """从 5 个关键点计算笑容分数（v4 相对基线）。"""
        eye_dist = np.sqrt((re_x - le_x) ** 2 + (re_y - le_y) ** 2)
        if eye_dist < 1:
            return 0, {"error": "eye_dist too small"}

        eye_avg_y = (re_y + le_y) / 2
        mouth_avg_y = (mr_y + ml_y) / 2
        eye_to_mouth_y_ratio = (mouth_avg_y - eye_avg_y) / eye_dist

        mouth_width = np.sqrt((mr_x - ml_x) ** 2 + (mr_y - ml_y) ** 2)
        mouth_width_ratio = mouth_width / eye_dist

        # 建立/使用基线
        if self._baseline is None or self._baseline_frames < 30:
            self._update_baseline(mouth_width_ratio, eye_to_mouth_y_ratio)
            # 前 30 帧：分数 = 0（让用户先稳定基线）
            return 0, {
                "mouth_width_ratio": mouth_width_ratio,
                "eye_to_mouth_y_ratio": eye_to_mouth_y_ratio,
                "baseline_ready": False,
                "baseline_frames": self._baseline_frames,
            }

        baseline_mw, baseline_em = self._baseline
        # 嘴部 Y 变小 = 大笑 (e2my 减小)
        em_diff = baseline_em - eye_to_mouth_y_ratio
        # 嘴宽增加 = 大笑
        mw_diff = mouth_width_ratio - baseline_mw

        # 经验值（基于你实测：露齿大笑 em_diff~0.02, mw_diff~0.05-0.10）:
        em_score = max(0, min(100, em_diff / 0.025 * 100))
        mw_score = max(0, min(100, mw_diff / 0.06 * 100))

        score = int((em_score + mw_score) / 2)

        return score, {
            "mouth_width_ratio": mouth_width_ratio,
            "eye_to_mouth_y_ratio": eye_to_mouth_y_ratio,
            "mouth_width_diff": mw_diff,
            "mouth_y_diff": em_diff,
            "baseline_mw": baseline_mw,
            "baseline_em": baseline_em,
            "em_score": em_score,
            "mw_score": mw_score,
        }

    def score(self, face_bgr: np.ndarray) -> Tuple[int, List[float]]:
        """兼容旧 API。"""
        return 0, [0.0] * 8
