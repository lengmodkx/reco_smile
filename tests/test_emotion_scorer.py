"""测试 EmotionScorer（v7 综合 max 算法）。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score


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


def test_closed_mouth_returns_low():
    """闭嘴分数应该 < 30"""
    scorer = EmotionScorer()
    # 闭嘴: extent=80, face_w=200, ratio=0.4, n2m=35, n2m_ratio=0.146
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=140, mr_y=215, ml_x=60, ml_y=215,  # extent=80
        face_w=200, face_h=240, nose_y=180,
    )
    assert info["mouth_extent_ratio"] < 0.50
    # 闭嘴时 score 应该是 0 (因为 n2m=0.146 < 0.16)
    assert score < 30, f"闭嘴应 < 30，实际 {score}"


def test_smile_returns_high():
    """露齿笑时 extent 上升，分数应该高"""
    scorer = EmotionScorer()
    # 露齿笑: extent_ratio=0.55 (上升), n2m 类似
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=155, mr_y=215, ml_x=45, ml_y=215,  # extent=110, ratio=0.55
        face_w=200, face_h=240, nose_y=180,
    )
    assert info["mouth_extent_ratio"] > 0.50
    assert score >= 50


def test_big_smile_returns_max():
    """大笑 score 应该 >= 60（v8 加权平均降低了分数上限）"""
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=165, mr_y=215, ml_x=35, ml_y=215,  # extent=130, ratio=0.65
        face_w=200, face_h=240, nose_y=180,
    )
    assert score >= 60, f"大笑应 >= 60，实际 {score}"


def test_invalid_face_w_returns_zero():
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100, le_x=50, le_y=100,
        mr_x=125, mr_y=215, ml_x=75, ml_y=215,
        face_w=0, face_h=240, nose_y=180,
    )
    assert score == 0
    assert "error" in info
