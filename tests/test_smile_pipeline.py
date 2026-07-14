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
    # 跳过模型加载（用 mock 替换）
    if not os.path.exists(config.YUNET_MODEL_PATH):
        monkeypatch.setattr(
            "core.smile_pipeline.FaceDetector", lambda *a, **kw: MagicMock()
        )
    if not os.path.exists(config.FERPLUS_MODEL_PATH):
        monkeypatch.setattr(
            "core.smile_pipeline.EmotionScorer", lambda *a, **kw: MagicMock()
        )

    pipeline = SmilePipeline(
        yunet_path=config.YUNET_MODEL_PATH,
        ferplus_path=config.FERPLUS_MODEL_PATH,
        photos_dir=str(tmp_path),
        threshold=60,
        cooldown=2.0,
    )
    # 注入 mock 的 detect/score 行为
    pipeline.face_detector.detect = MagicMock(return_value=[
        {"x": 100, "y": 100, "w": 200, "h": 200}
    ])
    pipeline.emotion_scorer.score = MagicMock(return_value=(85, [0.0]*8))
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
    pipeline_with_mocks.emotion_scorer.score = MagicMock(return_value=(80, [0.0]*8))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert result["saved_path"] is not None
    assert os.path.exists(result["saved_path"])


def test_pipeline_low_score_no_save(pipeline_with_mocks):
    """分数 < 阈值时不拍照"""
    pipeline_with_mocks.emotion_scorer.score = MagicMock(return_value=(40, [0.0]*8))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline_with_mocks.process(frame)
    assert result["saved_path"] is None


def test_pipeline_cooldown_prevents_consecutive_saves(pipeline_with_mocks):
    """冷却期内即使分数高也不重复拍"""
    pipeline_with_mocks.emotion_scorer.score = MagicMock(return_value=(80, [0.0]*8))
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
    # 人脸框区域应该有非灰色像素（绿色框）
    # 在 (100,100) 到 (300,300) 区域检查
    region = annotated[100:300, 100:300]
    assert region.std() > 5  # 框周围像素有变化
