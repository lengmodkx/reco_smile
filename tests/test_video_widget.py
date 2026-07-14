"""测试 VideoWidget：显示 BGR ndarray。"""
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_widget_creation(qt_app):
    from ui.video_widget import VideoWidget
    widget = VideoWidget()
    assert widget is not None


def test_update_frame_does_not_crash(qt_app):
    from ui.video_widget import VideoWidget
    widget = VideoWidget()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    widget.update_frame(frame)  # 不报错
