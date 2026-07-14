"""照片保存：按规则生成路径，编码 JPEG。"""
import os
from datetime import datetime
from typing import Optional
import cv2
import numpy as np


class PhotoSaver:
    """按日期目录 + 时间戳分数文件名保存 JPEG 照片。"""

    def __init__(self, photos_dir: str, jpeg_quality: int = 95):
        self.photos_dir = photos_dir
        self.jpeg_quality = jpeg_quality

    def generate_path(self, score: int, now: Optional[datetime] = None) -> str:
        """生成保存路径（不实际写文件）。"""
        if now is None:
            now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        ms_str = f"{now.microsecond // 1000:03d}"
        score_str = f"{score:02d}"
        filename = f"smile_{time_str}_{ms_str}_score{score_str}.jpg"
        return os.path.join(self.photos_dir, date_str, filename)

    def save(
        self, frame: np.ndarray, score: int, now: Optional[datetime] = None
    ) -> Optional[str]:
        """保存帧到磁盘。返回保存路径，失败返回 None。

        如果同名文件已存在，追加 _1, _2... 避免覆盖。
        """
        base_path = self.generate_path(score, now)
        directory = os.path.dirname(base_path)
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"[PhotoSaver] 创建目录失败: {e}")
            return None

        # 防覆盖：检查冲突并加后缀
        final_path = base_path
        if os.path.exists(final_path):
            stem, ext = os.path.splitext(base_path)
            counter = 1
            while os.path.exists(final_path):
                final_path = f"{stem}_{counter}{ext}"
                counter += 1

        try:
            cv2.imwrite(final_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return final_path
        except Exception as e:
            print(f"[PhotoSaver] 写入失败: {e}")
            return None
