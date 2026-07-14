"""测试 FaceDetector。
需要真实的 YuNet 模型文件才能跑大部分测试；这里先测试加载逻辑。
"""
import os
import pytest
import numpy as np
from core.face_detector import FaceDetector, ModelNotFoundError
import config


def test_load_raises_when_model_missing(tmp_path, monkeypatch):
    """模型文件不存在时应抛出 ModelNotFoundError"""
    fake_path = str(tmp_path / "nonexistent.onnx")
    with pytest.raises(ModelNotFoundError):
        FaceDetector(fake_path)


def test_detect_returns_list():
    """detect 应返回 list（可能为空）"""
    if not os.path.exists(config.YUNET_MODEL_PATH):
        pytest.skip("YuNet 模型文件不存在，跳过")
    detector = FaceDetector(config.YUNET_MODEL_PATH)
    # 创建一张全黑图作为测试输入（无人脸）
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)
    assert isinstance(faces, list)


def test_face_box_has_required_keys():
    """人脸框字典应包含 x, y, w, h 键"""
    if not os.path.exists(config.YUNET_MODEL_PATH):
        pytest.skip("YuNet 模型文件不存在，跳过")
    detector = FaceDetector(config.YUNET_MODEL_PATH)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)
    for face in faces:
        assert "x" in face
        assert "y" in face
        assert "w" in face
        assert "h" in face


def test_crop_face_returns_correct_size():
    """crop_face 应返回指定大小的彩色图像"""
    if not os.path.exists(config.YUNET_MODEL_PATH):
        pytest.skip("YuNet 模型文件不存在，跳过")
    detector = FaceDetector(config.YUNET_MODEL_PATH)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    face = {"x": 100, "y": 100, "w": 200, "h": 200}
    cropped = detector.crop_face(frame, face, output_size=(64, 64))
    assert cropped.shape == (64, 64, 3)
    assert cropped.dtype == np.uint8
