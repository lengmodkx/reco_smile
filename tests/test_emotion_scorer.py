"""测试 EmotionScorer：模型加载、推理、分数映射。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score, ModelNotFoundError
import config


# 分数映射算法测试（不需要模型）
# 算法：排除 p[0] baseline 和 p[1] non-face，重新归一化剩下 6 个类别
def test_smile_score_zero_for_no_happy_no_surprise():
    """不笑时（p[4] happiness=0，p[2] surprise=0）分数应为 0"""
    # 即使 p[0] = 0.8 baseline，happy=0 时分数仍应为 0
    probs = [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    assert compute_smile_score(probs, surprise_weight=0.3) == 0


def test_smile_score_100_for_all_happy():
    """全 happy（p[4]=1.0）时分数应为 100"""
    probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert compute_smile_score(probs, surprise_weight=0.3) == 100


def test_smile_score_combines_surprise():
    """惊喜加权：surprise 加入时分数更高"""
    # 场景 A: 0.3 happy, 0.0 surprise, 0.7 sad (排除 p[0]=0)
    #   remaining=1.0, renormalized happy = 0.3, score = 30
    # 场景 B: 0.3 happy, 0.5 surprise, 0.2 sad (排除 p[0]=0)
    #   remaining=1.0, renormalized happy = 0.3, surprise = 0.5
    #   score = (0.3 + 0.3*0.5) * 100 = 45 > 30
    probs_only_happy = [0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0]
    probs_happy_and_surprise = [0.0, 0.0, 0.5, 0.2, 0.3, 0.0, 0.0, 0.0]
    score_a = compute_smile_score(probs_only_happy, surprise_weight=0.3)
    score_b = compute_smile_score(probs_happy_and_surprise, surprise_weight=0.3)
    assert score_b > score_a, f"加入 surprise 应让分数更高：a={score_a}, b={score_b}"


def test_smile_score_capped_at_100():
    probs = [0.0, 0.0, 1.5, 0.0, 1.5, 0.0, 0.0, 0.0]
    assert compute_smile_score(probs, surprise_weight=0.3) == 100


def test_smile_score_different_weights():
    """权重越大，surprise 影响越大"""
    probs = [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    score_low = compute_smile_score(probs, surprise_weight=0.0)
    score_high = compute_smile_score(probs, surprise_weight=1.0)
    assert score_high > score_low


def test_smile_score_handles_high_baseline():
    """关键测试：即使 p[0] baseline 很高（0.85），只要 p[4] 上升也应能给出合理的分数"""
    # 模拟你的实际场景：p[0]=0.85 (中性 baseline), p[4]=0.10 (明显笑)
    # 重新归一化后 p[4] 占比 = 0.10 / (1-0.85-0) = 0.10/0.15 = 0.667 → 67 分
    probs = [0.85, 0.0, 0.0, 0.05, 0.10, 0.0, 0.0, 0.0]
    score = compute_smile_score(probs, surprise_weight=0.3)
    assert 60 <= score <= 80, f"应该给 60-80 分（baseline 高但 p[4] 上升），实际 {score}"


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
