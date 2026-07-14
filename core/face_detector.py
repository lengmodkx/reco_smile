"""人脸检测封装（基于 YuNet ONNX 模型）。
detect() 返回人脸框列表；crop_face() 裁剪并缩放到推理所需尺寸。
"""
import os
from typing import List, Dict, Tuple
import cv2
import numpy as np


class ModelNotFoundError(Exception):
    """模型文件不存在或加载失败时抛出"""


class FaceDetector:
    """YuNet 人脸检测器封装。"""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"人脸检测模型不存在: {model_path}\n"
                f"请按 assets/README.md 下载并放置模型文件。"
            )
        # OpenCV DNN 加载 YuNet
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),  # 任意值，会在 detect 时按实际帧大小调整
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5,
        )

    def detect(self, frame: np.ndarray) -> List[Dict[str, int]]:
        """检测人脸。返回 [{x, y, w, h}, ...]，无脸时返回 []。"""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        result = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            result.append({"x": x, "y": y, "w": w, "h": h})
        return result

    def crop_face(
        self, frame: np.ndarray, face: Dict[str, int], output_size: Tuple[int, int] = (64, 64)
    ) -> np.ndarray:
        """裁剪人脸区域并缩放到指定大小。返回 BGR ndarray。"""
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        # 边界保护
        x = max(0, x)
        y = max(0, y)
        x_end = min(frame.shape[1], x + w)
        y_end = min(frame.shape[0], y + h)
        cropped = frame[y:y_end, x:x_end]
        if cropped.size == 0:
            # 退化情况：返回灰色块而不是崩
            return np.full((output_size[1], output_size[0], 3), 128, dtype=np.uint8)
        resized = cv2.resize(cropped, output_size)
        return resized
