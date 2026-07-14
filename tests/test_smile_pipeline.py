"""测试 SmilePipeline：检测→打分→阈值→冷却→拍照的串联逻辑。"""
import os
import time
import numpy as np
import pytest
from unittest.mock import MagicMock
from core.smile_pipeline import SmilePipeline
import config


@pytest.fixture
def pipeline_with_mocks(tmp_path, monkeypatch):
    """构造一个使用 mock 的 pipeline，无需真实模型"""
    if not os.path.exists(config.YUNET_MODEL_PATH):
        monkeypatch.setattr(
            "core.smile_pipeline.FaceDetector", lambda *a, **kw: MagicMock()
        )

    pipeline = SmilePipeline(
        yunet_path=config.YUNET_MODEL_PATH,
        ferplus_path="",  # 不再使用
        photos_dir=str(tmp_path),
        threshold=60,
        cooldown=2.0,
    )
    # 注入 mock 的 detect 行为，返回含 landmarks 的人脸
    pipeline.face_detector.detect = MagicMock(return_value=[
        {
            "x": 100, "y": 100, "w": 200, "h": 200,
            "landmarks": {
                "right_eye": (300.0, 150.0),
                "left_eye": (150.0, 150.0),
                "nose": (220.0, 220.0),
                "mouth_right": (320.0, 280.0),
                "mouth_left": (180.0, 280.0),
            },
            "score": 0.95,
        }
    ])
    # mock 几何特征打分
    pipeline.emotion_scorer.score_from_landmarks = MagicMock(return_value=(85, {"mouth_width_ratio": 1.0}))
    return pipeline


def test_pipeline_processes_frame(pipeline_with_mocks):
    """process 返回 dict 包含必要字段"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert "score" in result
    assert "faces" in result
    assert "saved_path" in result
    assert "annotated_frame" in result


def test_pipeline_no_face_returns_zero(pipeline_with_mocks):
    """无人脸时分数为 0，不拍照"""
    pipeline_with_mocks.face_detector.detect = MagicMock(return_value=[])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert result["score"] == 0
    assert result["saved_path"] is None


def test_pipeline_high_score_triggers_save(pipeline_with_mocks):
    """分数 ≥ 阈值时应拍照"""
    pipeline_with_mocks.emotion_scorer.score_from_landmarks = MagicMock(return_value=(80, {}))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert result["saved_path"] is not None
    assert os.path.exists(result["saved_path"])


def test_pipeline_low_score_no_save(pipeline_with_mocks):
    """分数 < 阈值时不拍照"""
    pipeline_with_mocks.emotion_scorer.score_from_landmarks = MagicMock(return_value=(40, {}))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert result["saved_path"] is None


def test_pipeline_cooldown_prevents_consecutive_saves(pipeline_with_mocks):
    """冷却期内即使分数高也不重复拍"""
    pipeline_with_mocks.emotion_scorer.score_from_landmarks = MagicMock(return_value=(80, {}))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r1 = pipeline_with_mocks.process(frame)
    r2 = pipeline_with_mocks.process(frame)
    assert r1["saved_path"] is not None
    assert r2["saved_path"] is None  # 冷却期内


def test_pipeline_annotated_frame_has_face_box(pipeline_with_mocks):
    """annotated_frame 应是画了人脸框的 BGR 图像"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    result = pipeline_with_mocks.process(frame)
    annotated = result["annotated_frame"]
    assert annotated.shape == frame.shape
    region = annotated[100:300, 100:300]
    assert region.std() > 5
