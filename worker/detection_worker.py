"""后台工作线程：持续读摄像头帧、调用 pipeline、发射信号。"""
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.camera import Camera, CameraOpenError, FrameReadError
from core.smile_pipeline import SmilePipeline


class DetectionWorker(QThread):
    """在独立线程中运行摄像头 + 推理循环，通过信号把结果发回 UI 线程。"""

    # 信号：每处理完一帧发射一次
    # 参数：{"score", "faces", "saved_path", "annotated_frame", "fps"}
    frame_processed = pyqtSignal(dict)

    # 信号：摄像头或模型出错
    error_occurred = pyqtSignal(str)

    # 信号：Worker 启动完成（包括模型加载）
    started_signal = pyqtSignal()

    def __init__(
        self,
        yunet_path: str,
        ferplus_path: str,
        photos_dir: str,
        threshold: int = 60,
        cooldown: float = 3.0,
        camera_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._yunet_path = yunet_path
        self._ferplus_path = ferplus_path
        self._photos_dir = photos_dir
        self._threshold = threshold
        self._cooldown = cooldown
        self._camera_index = camera_index
        self._stop_requested = False

    def stop(self):
        """请求停止循环。线程会安全退出。"""
        self._stop_requested = True

    def run(self):
        """QThread 入口。"""
        try:
            # 初始化 pipeline（加载模型）
            pipeline = SmilePipeline(
                yunet_path=self._yunet_path,
                ferplus_path=self._ferplus_path,
                photos_dir=self._photos_dir,
                threshold=self._threshold,
                cooldown=self._cooldown,
            )
            camera = Camera(index=self._camera_index)
            camera.open()
            self.started_signal.emit()

            frame_count = 0
            fps_start = time.time()

            while not self._stop_requested:
                try:
                    frame = camera.read()
                except FrameReadError:
                    continue  # 跳过本帧

                result = pipeline.process(frame)
                frame_count += 1
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                result["fps"] = fps

                self.frame_processed.emit(result)

            camera.release()

        except CameraOpenError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(f"未知错误: {e}")
