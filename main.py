"""程序入口：启动 QApplication 和主窗口。

使用：
    python main.py
"""
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

import config
from ui.main_window import MainWindow


def check_models():
    """启动前检查模型文件是否存在。"""
    import os
    missing = []
    if not os.path.exists(config.YUNET_MODEL_PATH):
        missing.append(config.YUNET_MODEL_PATH)
    if not os.path.exists(config.FERPLUS_MODEL_PATH):
        missing.append(config.FERPLUS_MODEL_PATH)
    return missing


def main():
    app = QApplication(sys.argv)

    # 模型文件预检
    missing = check_models()
    if missing:
        msg = "以下模型文件缺失：\n\n"
        msg += "\n".join(f"  - {p}" for p in missing)
        msg += "\n\n请按 assets/README.md 下载模型后重试。"
        QMessageBox.critical(None, "模型缺失", msg)
        return 1

    window = MainWindow(
        yunet_path=config.YUNET_MODEL_PATH,
        ferplus_path=config.FERPLUS_MODEL_PATH,
        photos_dir=config.PHOTOS_DIR,
    )
    window.show()

    # 确保 photos 目录存在
    import os
    os.makedirs(config.PHOTOS_DIR, exist_ok=True)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
