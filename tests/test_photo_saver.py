"""测试 PhotoSaver：路径生成、JPEG 保存。"""
import os
import re
from datetime import datetime
import numpy as np
import pytest
from core.photo_saver import PhotoSaver


@pytest.fixture
def saver(tmp_path, monkeypatch):
    """使用临时目录作为照片根目录"""
    return PhotoSaver(photos_dir=str(tmp_path), jpeg_quality=95)


def test_generate_filename_includes_score_and_time(saver):
    path = saver.generate_path(score=87, now=datetime(2026, 7, 14, 15, 30, 45, 123000))
    filename = os.path.basename(path)
    # 期望格式: smile_153045_123_score87.jpg
    assert re.match(r"^smile_\d{6}_\d{3}_score\d{2}\.jpg$", filename), \
        f"文件名不符合规则: {filename}"


def test_generate_filename_in_date_subdir(saver):
    path = saver.generate_path(score=50, now=datetime(2026, 7, 14, 15, 30, 45, 0))
    assert "2026-07-14" in path


def test_save_creates_directory(saver):
    """保存时应自动创建日期子目录"""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    path = saver.save(frame, score=70, now=datetime(2026, 7, 14, 15, 30, 45, 0))
    assert os.path.exists(path)


def test_save_writes_valid_jpeg(saver):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    path = saver.save(frame, score=85, now=datetime(2026, 7, 14, 15, 30, 45, 0))
    # JPEG 文件应以 FFD8FF 开头
    with open(path, "rb") as f:
        header = f.read(3)
    assert header == b"\xff\xd8\xff", f"不是有效 JPEG: {header}"


def test_save_avoids_overwrite_with_suffix(saver):
    """同名文件已存在时应追加 _1, _2"""
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    now = datetime(2026, 7, 14, 15, 30, 45, 0)
    path1 = saver.save(frame, score=80, now=now)
    path2 = saver.save(frame, score=80, now=now)
    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)
