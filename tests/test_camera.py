"""测试 Camera 封装。
注意：真实摄像头测试需要硬件。这里只测试接口定义和无摄像头的错误处理。
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.camera import Camera, CameraOpenError, FrameReadError


def test_camera_open_raises_when_no_device():
    """当摄像头索引无效时，open 应该抛出 CameraOpenError"""
    cam = Camera(index=999)  # 大概率无效的索引
    with pytest.raises(CameraOpenError):
        cam.open()


def test_camera_read_returns_ndarray_after_open():
    """open 成功后，read 应返回 ndarray"""
    cam = Camera(index=0)
    try:
        cam.open()
    except CameraOpenError:
        pytest.skip("无摄像头设备，跳过此测试")
    frame = cam.read()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3  # H, W, C
    assert frame.shape[2] == 3  # BGR 三通道


def test_camera_release_is_idempotent():
    """release 可以多次调用，不报错"""
    cam = Camera(index=0)
    cam.release()  # 未 open 也能 release
    cam.release()  # 重复 release


def test_camera_context_manager():
    """支持 with 语句自动管理资源"""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        with Camera(index=0) as cam:
            frame = cam.read()
            assert frame is not None
        # with 退出后 release 应被调用
        mock_cap.release.assert_called()
