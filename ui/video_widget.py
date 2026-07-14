"""视频预览控件：把 BGR ndarray 实时显示到 QWidget。"""
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    """显示摄像头帧 + 人脸框 + 分数（已被 pipeline annotate 好）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        self.setText("Initializing...")

    def update_frame(self, bgr_frame: np.ndarray):
        """更新显示。BGR ndarray（OpenCV 格式）。"""
        h, w, ch = bgr_frame.shape
        bytes_per_line = ch * w
        # BGR → RGB
        rgb = bgr_frame[..., ::-1].copy()
        qt_image = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        # 等比缩放填充
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
