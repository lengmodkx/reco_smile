"""主窗口：组装 UI、连接 Worker 信号、处理关闭事件。"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

import config
from ui.video_widget import VideoWidget
from worker.detection_worker import DetectionWorker


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(self, yunet_path: str, ferplus_path: str, photos_dir: str):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        # 中心控件：视频预览
        self.video_widget = VideoWidget()
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_widget)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.set_status("Initializing...")

        # 启动 Worker
        self.worker = DetectionWorker(
            yunet_path=yunet_path,
            ferplus_path=ferplus_path,
            photos_dir=photos_dir,
            threshold=config.SMILE_THRESHOLD,
            cooldown=config.COOLDOWN_SECONDS,
            camera_index=config.CAMERA_INDEX,
        )
        self.worker.frame_processed.connect(self.handle_frame)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.started_signal.connect(
            lambda: self.set_status("Running")
        )
        self.worker.start()

    def set_status(self, message: str):
        """更新状态栏文字。"""
        self.status_bar.showMessage(message)

    def handle_frame(self, result: dict):
        """Worker 回调：更新画面和状态栏。"""
        self.video_widget.update_frame(result["annotated_frame"])
        score = result["score"]
        faces_count = len(result["faces"])
        fps = result.get("fps", 0)
        saved = result.get("saved_path")
        if faces_count == 0:
            msg = f"Running | FPS: {fps:.1f} | No face detected"
        else:
            msg = f"Running | FPS: {fps:.1f} | Faces: {faces_count} | Score: {score}"
        if saved:
            msg += f" | Saved: {saved}"
        self.set_status(msg)

    def handle_error(self, error_msg: str):
        """Worker 错误回调：状态栏显示错误。"""
        self.set_status(f"Error: {error_msg}")

    def closeEvent(self, event: QCloseEvent):
        """窗口关闭时清理 Worker。"""
        self.set_status("Stopping...")
        self.worker.stop()
        if not self.worker.wait(2000):  # 2 秒超时
            self.worker.terminate()
            self.worker.wait()
        event.accept()
