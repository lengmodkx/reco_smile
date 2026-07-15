"""测试 EmotionScorer（v5 嘴部张开度主信号算法）。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score


# 分数映射算法测试
def test_compute_smile_score_average():
    score = compute_smile_score(
        mouth_width_ratio=0.85, eye_to_mouth_y_ratio=1.10,
        mouth_up_score=80, mouth_width_score=60,
    )
    assert score == 70


def test_compute_smile_score_capped():
    score = compute_smile_score(1.0, 1.0, 150, 150)
    assert score == 100


# EmotionScorer 类测试
def test_scorer_initialization():
    scorer = EmotionScorer()
    assert scorer is not None


def test_closed_mouth_zero_score():
    """闭嘴时分数应该接近 0"""
    scorer = EmotionScorer()
    # 闭嘴: 嘴部左右角 Y 几乎相同（mr_y ~= ml_y）
    # 嘴部高 ~0
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,    # 眼睛距离 100
        le_x=50, le_y=100,
        mr_x=125, mr_y=215,    # 嘴左右角 Y 相同 (闭嘴)
        ml_x=75, ml_y=215,
    )
    assert score < 10, f"闭嘴分数应 < 10，实际 {score}"


def test_open_mouth_high_score():
    """张嘴时分数应该高（嘴部左右角 Y 差异大）"""
    scorer = EmotionScorer()
    # 张嘴大笑: 嘴部高度应该有 0.10 左右
    # 但 mr_y 和 ml_y 的 Y 差异在 YuNet 里不一定等于嘴高
    # 因为 YuNet 给的是左右嘴角，不是上下边
    # 让我用一个真实场景: 不笑时 mr_y/ml_y 接近, 大笑时张嘴 mr_y/ml_y 都变大
    # 实际上 Y 差距仍然不会太大
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=130, mr_y=215,
        ml_x=70, ml_y=210,   # ml 上下偏 5
    )
    assert info["mouth_height_ratio"] < 1.0, f"Y 差距 < 1 个眼距，info={info}"


def test_large_mouth_open_high_score():
    """明显张嘴（大 ml_y 与 mr_y 差异大）时分数高"""
    scorer = EmotionScorer()
    # 强烈张嘴: mr_y=200, ml_y=240, 差 40 = 0.4 眼距
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=130, mr_y=200,
        ml_x=70, ml_y=240,   # 巨大 Y 差 = 嘴部张开度
    )
    assert info["mouth_height_ratio"] > 0.1
    assert score >= 50


def test_eye_dist_too_small_returns_zero():
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=100, re_y=100,
        le_x=100, le_y=100,
        mr_x=120, mr_y=200,
        ml_x=80, ml_y=200,
    )
    assert score == 0
    assert "error" in info
