"""测试 EmotionScorer：模型加载、推理、分数映射。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score, ModelNotFoundError
import config


# 分数映射算法测试（不需要模型）
def test_smile_score_zero_for_no_happy_no_surprise():
    probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 全 neutral
    assert compute_smile_score(probs, surprise_weight=0.3) == 0


def test_smile_score_100_for_all_happy():
    probs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 全 happy
    assert compute_smile_score(probs, surprise_weight=0.3) == 100


def test_smile_score_combines_surprise():
    """惊喜加权：大笑时分数应明显高于纯微笑"""
    probs_only_happy = [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]
    probs_happy_and_surprise = [0.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    score_a = compute_smile_score(probs_only_happy, surprise_weight=0.3)
    score_b = compute_smile_score(probs_happy_and_surprise, surprise_weight=0.3)
    assert score_b > score_a


def test_smile_score_capped_at_100():
    probs = [0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # 概率和 > 1
    assert compute_smile_score(probs, surprise_weight=0.3) == 100


def test_smile_score_different_weights():
    """权重越大，surprise 影响越大"""
    probs = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    score_low = compute_smile_score(probs, surprise_weight=0.0)
    score_high = compute_smile_score(probs, surprise_weight=1.0)
    assert score_high > score_low


# 模型加载测试（需要真实模型）
def test_load_raises_when_model_missing(tmp_path):
    fake_path = str(tmp_path / "nonexistent.onnx")
    with pytest.raises(ModelNotFoundError):
        EmotionScorer(fake_path)


def test_score_returns_int_and_probs():
    if not os.path.exists(config.FERPLUS_MODEL_PATH):
        pytest.skip("FerPlus 模型文件不存在，跳过")
    scorer = EmotionScorer(config.FERPLUS_MODEL_PATH)
    # 64x64 BGR 图像（FerPlus 期望输入）
    dummy_face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    score, probs = scorer.score(dummy_face)
    assert isinstance(score, int)
    assert 0 <= score <= 100
    assert isinstance(probs, list)
    assert len(probs) == 8
    assert abs(sum(probs) - 1.0) < 0.01  # softmax 概率和约等于 1
