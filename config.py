"""全局配置常量。修改这里的值即可调整应用行为，无需改其他代码。"""
import os

# ============ 摄像头 ============
CAMERA_INDEX = 0           # 系统默认摄像头
CAMERA_WIDTH = 640         # 摄像头分辨率宽
CAMERA_HEIGHT = 480        # 摄像头分辨率高

# ============ 笑容识别 ============
SMILE_THRESHOLD = 25       # 分数 ≥ 此值触发拍照（0-100）- 调低以适配 YuNet 5 关键点的精度
COOLDOWN_SECONDS = 3.0     # 拍照后冷却期（秒），避免连拍
SURPRISE_WEIGHT = 0.3      # surprise 概率在分数中的权重（不再使用，保留兼容）

# ============ 模型路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
YUNET_MODEL_PATH = os.path.join(ASSETS_DIR, "face_detection_yunet.onnx")
FERPLUS_MODEL_PATH = os.path.join(ASSETS_DIR, "emotion-ferplus-8.onnx")

# ============ 存储 ============
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")  # 照片根目录
JPEG_QUALITY = 95                                # JPEG 编码质量

# ============ UI ============
WINDOW_TITLE = "SmileScore"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
STATUS_BAR_HEIGHT = 30
