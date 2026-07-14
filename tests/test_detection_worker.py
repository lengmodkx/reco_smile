"""测试 DetectionWorker：QThread 生命周期、信号发射。"""
import os
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

# 必须有 QApplication 才能用 QThread
@pytest.fixture(scope="module")
def qt_app():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


@pytest.fixture
def worker(qt_app, tmp_path, monkeypatch):
    """构造一个使用 mock pipeline 的 worker"""
    # mock 掉 pipeline 创建，避免加载真实模型
    mock_pipeline = MagicMock()
    mock_pipeline.process = MagicMock(return_value={
        "score": 75,
        "faces": [{"x": 100, "y": 100, "w": 200, "h": 200}],
        "saved_path": None,
        "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
    })
    monkeypatch.setattr(
        "worker.detection_worker.SmilePipeline", lambda **kw: mock_pipeline
    )
    monkeypatch.setattr(
        "worker.detection_worker.Camera", lambda index=0: MagicMock()
    )

    from worker.detection_worker import DetectionWorker
    return DetectionWorker(
        yunet_path="fake.onnx",
        ferplus_path="fake.onnx",
        photos_dir=str(tmp_path),
        threshold=60,
        cooldown=3.0,
    )


def test_worker_emits_frame_processed_signal(qt_app, worker):
    """worker 启动后应周期性发射 frame_processed 信号"""
    received = []
    worker.frame_processed.connect(lambda d: received.append(d))

    worker.start()
    # 给 worker 200ms 运行
    loop = QEventLoop()
    QTimer.singleShot(200, loop.quit)
    loop.exec()
    worker.stop()
    worker.wait(2000)

    assert len(received) > 0, "worker 启动后未收到任何信号"


def test_worker_stop_is_idempotent(qt_app, worker):
    """stop 可多次调用不报错"""
    worker.stop()
    worker.stop()
