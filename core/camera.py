"""摄像头封装：打开、读帧、释放。失败时抛出明确异常。"""
from typing import Optional
import cv2
import numpy as np


class CameraOpenError(Exception):
    """摄像头无法打开时抛出"""


class FrameReadError(Exception):
    """帧读取失败时抛出"""


class Camera:
    """系统摄像头的薄封装。"""

    def __init__(self, index: int = 0):
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """打开摄像头。失败抛出 CameraOpenError。"""
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap = None
            raise CameraOpenError(
                f"无法打开摄像头 (index={self.index})。"
                f"请检查设备是否连接，或关闭其他占用摄像头的应用。"
            )

    def read(self) -> np.ndarray:
        """读取一帧。返回 BGR ndarray。失败抛出 FrameReadError。"""
        if self._cap is None:
            raise FrameReadError("摄像头未打开，请先调用 open()")
        ret, frame = self._cap.read()
        if not ret:
            raise FrameReadError("帧读取失败（摄像头可能已断开）")
        return frame

    def release(self) -> None:
        """释放摄像头资源。可多次调用。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
