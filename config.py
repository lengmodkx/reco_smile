"""项目配置"""
import os

# 保存抓拍照片的根目录
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured")

# 摄像头索引，0 是默认摄像头
CAMERA_INDEX = 0

# 画面分辨率（可根据摄像头支持调整）
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 笑容分数阈值，超过此值且持续指定帧数后拍照
SMILE_THRESHOLD = 20

# 连续满足阈值的帧数才触发拍照（防抖）
SMILE_CONSECUTIVE_FRAMES = 3

# 拍照后冷却帧数，避免连续拍多张
CAPTURE_COOLDOWN_FRAMES = 30

# 是否在画面上显示调试信息（人脸框、关键点、分数）
SHOW_DEBUG_INFO = True

# 是否镜像显示（自拍习惯）
MIRROR_PREVIEW = True
