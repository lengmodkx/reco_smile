"""测试 EmotionScorer（几何特征版）。"""
import os
import numpy as np
import pytest
from core.emotion_scorer import EmotionScorer, compute_smile_score


# 分数映射算法测试（不需要模型）
def test_smile_score_zero_when_mouth_very_narrow():
    """嘴很窄（嘴角下压）时分数应 < 20"""
    # 0.60 对应 0 分
    score = compute_smile_score(mouth_width_ratio=0.60)
    assert score < 20, f"嘴很窄应该分数很低，实际 {score}"


def test_smile_score_hundred_for_very_wide_mouth():
    """嘴很宽（大笑）时分数应 ≥ 90"""
    score = compute_smile_score(mouth_width_ratio=1.20)
    assert score >= 90, f"嘴很宽应该分数 ≥ 90，实际 {score}"


def test_smile_score_mid_range():
    """嘴宽中等时分数应在 30-70"""
    score = compute_smile_score(mouth_width_ratio=0.85)
    assert 30 <= score <= 70, f"中等嘴宽应该分数 30-70，实际 {score}"


def test_smile_score_capped_at_100():
    """分数最大 100"""
    score = compute_smile_score(mouth_width_ratio=2.0)
    assert score == 100


def test_smile_score_with_mouth_open_bonus():
    """张嘴有 bonus 加成"""
    score_closed = compute_smile_score(mouth_width_ratio=0.85, eye_to_mouth_y_ratio=0.85)
    score_open = compute_smile_score(mouth_width_ratio=0.85, eye_to_mouth_y_ratio=0.65)
    assert score_open > score_closed, f"张嘴分数应更高: closed={score_closed}, open={score_open}"


# EmotionScorer 类测试
def test_scorer_initialization():
    """不再需要 ferplus 模型，构造应不报错"""
    scorer = EmotionScorer()
    assert scorer is not None


def test_score_from_landmarks_with_neutral_face():
    """中性脸：眼睛距离 200px，嘴宽 160px（嘴宽/眼距=0.8）"""
    scorer = EmotionScorer()
    score, info = scorer.score_from_landmarks(
        re_x=400, re_y=200,  # 右眼
        le_x=200, le_y=200,  # 左眼
        mr_x=350, mr_y=320,  # 嘴右
        ml_x=250, ml_y=320,  # 嘴左
    )
    # 眼距 = 200, 嘴宽 = 100, ratio = 0.5
    assert "mouth_width_ratio" in info
    assert 0 < score < 100


def test_score_from_landmarks_with_wide_smile():
    """明显笑：嘴宽/眼距 > 1.0"""
    scorer = EmotionScorer()
    # 眼睛距离 100，嘴宽 120（大笑）
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=140, mr_y=180,
        ml_x=20, ml_y=180,
    )
    assert info["mouth_width_ratio"] > 1.0
    assert score >= 80, f"大笑应该分数 ≥ 80，实际 {score}"


def test_score_from_landmarks_with_frown():
    """嘴角下压：嘴宽/眼距 < 0.7"""
    scorer = EmotionScorer()
    # 眼睛距离 100，嘴宽 50（嘴角下压）
    score, info = scorer.score_from_landmarks(
        re_x=150, re_y=100,
        le_x=50, le_y=100,
        mr_x=100, mr_y=200,  # 嘴右位置内收
        ml_x=60, ml_y=200,  # 嘴左位置内收
    )
    assert info["mouth_width_ratio"] < 0.7
    assert score < 30, f"嘴角下压应该分数 < 30，实际 {score}"
