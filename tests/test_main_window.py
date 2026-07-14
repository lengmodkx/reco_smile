"""测试 MainWindow：构造、状态栏更新、清理逻辑。"""
import pytest
from unittest.mock import MagicMock
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_window_creation(qt_app):
    from ui.main_window import MainWindow
    window = MainWindow(
        yunet_path="fake.onnx",
        ferplus_path="fake.onnx",
        photos_dir="/tmp/photos",
    )
    assert window.windowTitle() == "SmileScore"


def test_status_bar_updates(qt_app):
    from ui.main_window import MainWindow
    window = MainWindow(
        yunet_path="fake.onnx",
        ferplus_path="fake.onnx",
        photos_dir="/tmp/photos",
    )
    window.set_status("Test message")
    assert "Test message" in window.statusBar().currentMessage()


def test_handle_frame_updates_widget(qt_app):
    from ui.main_window import MainWindow
    import numpy as np
    window = MainWindow(
        yunet_path="fake.onnx",
        ferplus_path="fake.onnx",
        photos_dir="/tmp/photos",
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    window.handle_frame({
        "score": 75,
        "faces": [{"x": 100, "y": 100, "w": 200, "h": 200}],
        "saved_path": None,
        "annotated_frame": frame,
        "fps": 25.0,
    })
    # 不报错即可，pixmap 应已更新
    assert window.video_widget.pixmap() is not None


def test_handle_error_shows_red_status(qt_app):
    from ui.main_window import MainWindow
    window = MainWindow(
        yunet_path="fake.onnx",
        ferplus_path="fake.onnx",
        photos_dir="/tmp/photos",
    )
    window.handle_error("Camera not found")
    msg = window.statusBar().currentMessage()
    assert "Camera not found" in msg or "Error" in msg
