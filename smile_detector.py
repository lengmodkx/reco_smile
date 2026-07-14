"""
笑容检测器：基于 MediaPipe Face Mesh 的人脸关键点，计算 0~100 的笑容分数。
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class SmileResult:
    """检测结果"""
    score: float                      # 0~100 的笑容分数
    face_box: Optional[Tuple[int, int, int, int]]  # 人脸框 (x, y, w, h)
    landmarks: Optional[List[Tuple[int, int]]]     # 人脸关键点（像素坐标）
    angle: float                      # 嘴角上扬角度（度）
    mouth_open_ratio: float           # 嘴张开比例


class SmileDetector:
    """
    基于 MediaPipe Face Mesh 的笑容检测器。

    核心思路：
    1. 用 MediaPipe Face Mesh 提取 468 个人脸关键点。
    2. 取嘴唇相关点：左嘴角 61、右嘴角 291、上唇中心 0、下唇中心 17。
    3. 计算嘴角上扬角度和嘴张开比例。
    4. 将几何特征映射为 0~100 的笑容分数。
    """

    # MediaPipe Face Mesh 嘴唇关键点点号
    LEFT_CORNER = 61
    RIGHT_CORNER = 291
    UPPER_LIP = 0
    LOWER_LIP = 17

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Optional[SmileResult]:
        """
        对单帧图像进行笑容检测。

        Args:
            frame: BGR 图像，numpy array。

        Returns:
            SmileResult 或 None（未检测到人脸）。
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        # 只取第一张脸
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # 将归一化坐标转为像素坐标
        def to_px(idx: int) -> Tuple[int, int]:
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        # 一次性把 468 个关键点转成像素坐标，后续从中按点号取用，避免重复计算
        all_px = [to_px(i) for i in range(len(landmarks))]
        left_corner = all_px[self.LEFT_CORNER]
        right_corner = all_px[self.RIGHT_CORNER]
        upper_lip = all_px[self.UPPER_LIP]
        lower_lip = all_px[self.LOWER_LIP]

        # 计算嘴宽（左右嘴角距离）
        mouth_width = self._distance(left_corner, right_corner)
        if mouth_width < 1e-6:
            mouth_width = 1e-6

        # 计算嘴张开高度（上下唇中心距离）
        mouth_height = self._distance(upper_lip, lower_lip)

        # 嘴张开比例：高度 / 宽度
        mouth_open_ratio = mouth_height / mouth_width

        # 嘴角相对唇心的上扬量（归一化到嘴宽）：
        # >0 表示嘴角高于唇心（上扬/微笑），<0 表示下垂。
        # 注意 y 轴向下为正，所以“上扬”对应 y 更小。
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2.0
        corners_avg_y = (left_corner[1] + right_corner[1]) / 2.0
        curvature = (mouth_center_y - corners_avg_y) / mouth_width

        # 嘴角连线相对水平线的倾角（仅用于调试展示，不参与评分）
        angle = math.degrees(
            math.atan2(right_corner[1] - left_corner[1], right_corner[0] - left_corner[0])
        )

        # 映射为 0~100 分数
        score = self._compute_score(curvature, mouth_open_ratio)

        # 计算人脸框（用于可视化）——单次遍历（numpy）求包围盒
        pts = np.asarray(all_px)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        face_box = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

        return SmileResult(
            score=score,
            face_box=face_box,
            landmarks=all_px,
            angle=angle,
            mouth_open_ratio=mouth_open_ratio,
        )

    def release(self) -> None:
        """释放 MediaPipe 资源。"""
        self.face_mesh.close()

    @staticmethod
    def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _compute_score(curvature: float, mouth_open_ratio: float) -> float:
        """
        根据嘴角上扬程度和嘴张开比例计算笑容分数。

        参数说明：
        - curvature: 嘴角相对唇心的上扬量，已归一化到嘴宽。
          >0 表示嘴角上扬（微笑），=0 表示水平，<0 表示下垂。
          该指标对“对称微笑”同样有效——不像旧的“嘴角连线倾角”，
          后者只有在左右嘴角高度不一致时才非零，对称笑容会被算成 0 分。
        - mouth_open_ratio: 嘴张开高度 / 嘴宽度。

        映射规则（经验值，可按实际调整）：
        - 上扬量 curvature：0.00 映射为 0 分，0.12 映射为 100 分。
        - 嘴张开比例：0.40 映射为 0 分，0.55 映射为 100 分。
        - 最终分数 = 0.85 * 上扬分 + 0.15 * 张开分。
        """
        # 嘴角上扬分数
        curve_score = curvature / 0.12 * 100.0
        curve_score = max(0.0, min(100.0, curve_score))

        # 嘴张开分数
        open_score = (mouth_open_ratio - 0.40) / 0.15 * 100.0
        open_score = max(0.0, min(100.0, open_score))

        # 加权综合：嘴角上扬是笑容的主要特征，嘴张开度作为辅助
        final_score = 0.85 * curve_score + 0.15 * open_score
        return round(final_score, 1)


# 便于命令行快速测试
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python smile_detector.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)

    detector = SmileDetector(static_image_mode=True)
    try:
        result = detector.detect(img)
        if result is None:
            print("未检测到人脸")
        else:
            print(f"笑容分数: {result.score}")
            print(f"嘴角角度: {result.angle:.2f}°")
            print(f"嘴张开比: {result.mouth_open_ratio:.3f}")
    finally:
        detector.release()
