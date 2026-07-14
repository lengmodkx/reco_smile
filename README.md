# SmileScore

跨平台离线桌面应用，使用摄像头实时识别人脸表情，按 0-100 打分笑容强度，分数超过阈值时自动拍照保存到本地。

## 环境要求

- Python 3.10 或更高版本
- 操作系统：Windows / Linux / macOS
- 一个可用的摄像头

## 安装

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型文件（手动）
# 见 assets/README.md
```

## 模型下载

下载到 `assets/` 目录：

1. **YuNet 人脸检测**（必需）: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

**注意**：本项目已弃用 FerPlus 表情模型，改用 YuNet 自带的 5 个面部关键点 + 几何特征打分（嘴宽/眼距 + 嘴部 Y 偏移）。

## 运行

```bash
python main.py
```

## 打包成 Windows exe

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "assets;assets" main.py
# 产物在 dist/main.exe
```

## 常见问题

**Q: 摄像头打不开？**
A: 检查系统设置中摄像头权限；关闭其他占用摄像头的应用（如 Zoom、Teams）。

**Q: 模型加载失败？**
A: 确认 `assets/` 目录下有两个 .onnx 文件，文件名与上面链接一致。

**Q: FPS 太低？**
A: 降低摄像头分辨率（修改 `config.py` 中的 `CAMERA_WIDTH`）。
