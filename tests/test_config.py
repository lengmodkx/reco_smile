"""测试配置常量的合理性"""
from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    SMILE_THRESHOLD,
    COOLDOWN_SECONDS,
    YUNET_MODEL_PATH,
    FERPLUS_MODEL_PATH,
    PHOTOS_DIR,
    JPEG_QUALITY,
    SURPRISE_WEIGHT,
)


def test_camera_index_is_int():
    assert isinstance(CAMERA_INDEX, int)
    assert CAMERA_INDEX >= 0


def test_camera_resolution_positive():
    assert CAMERA_WIDTH > 0
    assert CAMERA_HEIGHT > 0


def test_smile_threshold_in_range():
    assert 0 <= SMILE_THRESHOLD <= 100


def test_cooldown_positive():
    assert COOLDOWN_SECONDS > 0


def test_model_paths_exist():
    """路径字符串存在即可，文件不需要存在（用户后续手动下载）"""
    assert isinstance(YUNET_MODEL_PATH, str)
    assert len(YUNET_MODEL_PATH) > 0
    assert isinstance(FERPLUS_MODEL_PATH, str)
    assert len(FERPLUS_MODEL_PATH) > 0


def test_photos_dir_is_string():
    assert isinstance(PHOTOS_DIR, str)
    assert len(PHOTOS_DIR) > 0


def test_jpeg_quality_in_range():
    assert 1 <= JPEG_QUALITY <= 100


def test_surprise_weight_in_range():
    assert 0 <= SURPRISE_WEIGHT <= 1


def test_threshold_default_is_reasonable():
    """默认阈值应该不会太低（漏拍）也不会太高（不拍）"""
    # v6.1 后阈值降到 25 以适配 YuNet 5 关键点的精度
    assert 20 <= SMILE_THRESHOLD <= 80
