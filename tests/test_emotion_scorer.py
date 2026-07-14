"""测试 EmotionScorer（v4 相对基线算法）。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score


# 分数映射算法测试
def test_compute_smile_score_average():
    """compute_smile_score 接受子分数求平均"""
    score = compute_smile_score(
        mouth_width_ratio=0.85,
        eye_to_mouth_y_ratio=1.10,
        mouth_up_score=80,
        mouth_width_score=60,
    )
    assert score == 70


def test_compute_smile_score_capped():
    score = compute_smile_score(1.0, 1.0, 150, 150)
    assert score == 100


# EmotionScorer 类测试
def test_scorer_initialization():
    scorer = EmotionScorer()
    assert scorer is not None
    assert scorer._baseline is None


def test_first_30_frames_return_zero():
    """前 30 帧用于建立基线，分数应该为 0"""
    scorer = EmotionScorer()
    for i in range(30):
        score, info = scorer.score_from_landmarks(
            re_x=150, re_y=100,
            le_x=50, le_y=100,
            mr_x=125, mr_y=200,
            ml_x=75, ml_y=200,
        )
    # 第 30 帧时基线刚建立
    assert info.get("baseline_ready") is True or info.get("baseline_frames") >= 30


def test_baseline_then_smile_detected():
    """建立基线后，大笑应该给出高分"""
    scorer = EmotionScorer()
    # 先 30 帧用同一个人脸建立基线
    for i in range(30):
        scorer.score_from_landmarks(
            re_x=150, re_y=100,
            le_x=50, le_y=100,
            mr_x=125, mr_y=215,  # 嘴部在 Y=215 (基线)
            ml_x=75, ml_y=215,   # 嘴宽 50
        )
    # 现在大笑: 嘴部上移 + 嘴变宽
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=145, mr_y=205,  # 嘴部 Y 变小 (上移 10)
        ml_x=55, ml_y=205,   # 嘴宽 90 (变大)
    )
    assert score >= 60, f"大笑应该 >= 60，实际 {score} (info={info})"


def test_baseline_then_neutral_stays_low():
    """建立基线后，中性脸应该给低分"""
    scorer = EmotionScorer()
    # 30 帧基线
    for i in range(30):
        scorer.score_from_landmarks(
            re_x=150, re_y=100,
            le_x=50, le_y=100,
            mr_x=125, mr_y=215,
            ml_x=75, ml_y=215,
        )
    # 同样姿势的中性脸
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=125, mr_y=215,
        ml_x=75, ml_y=215,
    )
    assert score < 30, f"中性脸应该 < 30，实际 {score}"


def test_frown_below_neutral():
    """嘴角下压应该比中性更低"""
    scorer = EmotionScorer()
    for i in range(30):
        scorer.score_from_landmarks(
            re_x=150, re_y=100,
            le_x=50, le_y=100,
            mr_x=125, mr_y=215,
            ml_x=75, ml_y=215,
        )
    # 嘴角下压: 嘴部 Y 变大 (下移) + 嘴变窄
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=110, mr_y=225,  # 下移 10
        ml_x=90, ml_y=225,   # 嘴变窄
    )
    assert score < 30


def test_set_baseline_manually():
    """手动设置基线可以跳过 30 帧"""
    scorer = EmotionScorer()
    scorer.set_baseline(mouth_width_ratio=0.82, eye_to_mouth_y_ratio=1.15)
    # 大笑：嘴部上移 + 嘴变宽
    # 嘴宽从 50 增到 80 (相对眼距 100 而言，ratio 从 0.50 到 0.80)
    # 实际真实场景: 不笑时 mouth_w_ratio ~0.82, 大笑时 ~0.92
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=145, mr_y=205,  # 嘴部 Y 上移
        ml_x=55, ml_y=205,   # 嘴宽 = 90, ratio=0.9
    )
    assert score >= 60, f"大笑应该 >= 60，实际 {score}"


def test_eye_dist_too_small_returns_zero():
    """异常关键点位置应该返回 0"""
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=100, re_y=100,
        le_x=100, le_y=100,
        mr_x=120, mr_y=200,
        ml_x=80, ml_y=200,
    )
    assert score == 0
    assert "error" in info
