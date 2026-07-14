"""笑容识别流水线：检测 → 打分 → 阈值判断 → 冷却 → 拍照。
核心调度逻辑，UI 和 Worker 不直接调用底层模块，都通过本类。
"""
import time
from typing import Dict, Optional, Any
import numpy as np
import cv2

from core.face_detector import FaceDetector
from core.emotion_scorer import EmotionScorer
from core.photo_saver import PhotoSaver


class SmilePipeline:
    """串联各核心模块，对单帧做完整处理。"""

    def __init__(
        self,
        yunet_path: str,
        ferplus_path: str,
        photos_dir: str,
        threshold: int = 60,
        cooldown: float = 3.0,
        jpeg_quality: int = 95,
        surprise_weight: float = 0.3,
    ):
        self.face_detector = FaceDetector(yunet_path)
        self.emotion_scorer = EmotionScorer(ferplus_path, surprise_weight)
        self.photo_saver = PhotoSaver(photos_dir, jpeg_quality)
        self.threshold = threshold
        self.cooldown = cooldown
        self._last_save_time = 0.0  # epoch 秒

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理一帧，返回结果字典。

        Returns:
            {
                "score": int,              # 0-100
                "faces": list,             # [{x,y,w,h}, ...]
                "saved_path": str | None,  # 本帧是否触发了拍照
                "annotated_frame": ndarray # 画了人脸框的原帧副本
            }
        """
        annotated = frame.copy()
        faces = self.face_detector.detect(frame)

        # 画人脸框
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        score = 0
        saved_path: Optional[str] = None

        if not faces:
            return {
                "score": 0,
                "faces": [],
                "saved_path": None,
                "annotated_frame": annotated,
            }

        # 取最大的人脸打分（多脸场景）
        largest = max(faces, key=lambda f: f["w"] * f["h"])
        face_img = self.face_detector.crop_face(frame, largest)
        score, _ = self.emotion_scorer.score(face_img)

        # 在框上方写分数
        text = f"Score: {score}"
        cv2.putText(
            annotated,
            text,
            (largest["x"], largest["y"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 阈值判断 + 冷却检查
        now = time.time()
        if score >= self.threshold and (now - self._last_save_time) >= self.cooldown:
            saved_path = self.photo_saver.save(frame, score)
            if saved_path is not None:
                self._last_save_time = now

        return {
            "score": score,
            "faces": faces,
            "saved_path": saved_path,
            "annotated_frame": annotated,
        }
