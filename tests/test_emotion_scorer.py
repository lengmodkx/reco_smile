"""测试 EmotionScorer（v6 嘴角外推型算法）。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score


# compute_smile_score 兼容测试
def test_compute_smile_score_average():
    score = compute_smile_score(
        mouth_width_ratio=0.85, eye_to_mouth_y_ratio=1.10,
        mouth_up_score=80, mouth_width_score=60,
    )
    assert score == 70


def test_compute_smile_score_capped():
    score = compute_smile_score(1.0, 1.0, 150, 150)
    assert score == 100


def test_scorer_initialization():
    scorer = EmotionScorer()
    assert scorer is not None


def test_closed_mouth_zero_score():
    """闭嘴时嘴部 X 范围小，分数应该接近 0"""
    scorer = EmotionScorer()
    # face_w=200, 嘴角距离 = 90, ratio = 0.45 -> 0 分
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=145, mr_y=215,  # 嘴右
        ml_x=55, ml_y=215,   # 嘴左, 距离 = 90
        face_w=200, face_h=240,
        nose_y=180,
    )
    assert score < 20, f"闭嘴分数应低，实际 {score} (info={info})"


def test_smile_mouth_extended_score():
    """露齿笑时嘴角外推，分数应该高"""
    scorer = EmotionScorer()
    # face_w=200, 嘴角距离 = 130, ratio = 0.65 -> 100 分
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=165, mr_y=215,  # 嘴右角外推
        ml_x=35, ml_y=215,   # 嘴左角外推, 距离 = 130
        face_w=200, face_h=240,
        nose_y=180,
    )
    assert info["mouth_extent_ratio"] > 0.55
    assert score >= 60, f"露齿笑应 >= 60，实际 {score} (info={info})"


def test_no_smile_signal_returns_zero():
    """嘴部非常窄（闭嘴）应该低分"""
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=125, mr_y=215, ml_x=75, ml_y=215,  # 嘴部很窄
        face_w=200, face_h=240, nose_y=180,
    )
    assert info["mouth_extent_ratio"] < 0.30
    assert score < 20


def test_invalid_face_w_returns_zero():
    """face_w=0 应该返回 0"""
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=125, mr_y=215, ml_x=75, ml_y=215,
        face_w=0, face_h=240, nose_y=180,
    )
    assert score == 0
    assert "error" in info
