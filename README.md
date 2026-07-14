# 笑脸识别与抓拍 MVP

基于 Python + OpenCV + MediaPipe 的笑容识别程序。可调用摄像头实时检测人脸笑容并给出 0~100 的分数，分数达标后自动拍照保存。

## 功能

- 实时检测摄像头画面中最大的人脸。
- 根据**嘴角上扬角度**和**嘴张开比例**计算笑容分数（0 = 不笑，100 = 大笑）。
- 分数超过阈值且持续多帧后自动抓拍照片。
- 支持手动按空格键抓拍。
- 保存路径为 `./captured/`，文件名包含时间戳和分数。

## 环境要求

- Python 3.10+
- 摄像头（笔记本自带或 USB 摄像头）
- Windows / Linux / macOS 均可运行

## 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：`mediapipe` 版本需使用 `0.10.13`，新版（0.10.30+）的 API 结构有变化，当前代码不兼容。

## 运行主程序

```bash
python main.py
```

按键：

- `q` / `ESC`：退出程序
- `空格`：手动抓拍

## 项目结构

```
.
├── config.py              # 配置参数（阈值、摄像头索引、保存路径等）
├── main.py                # 主程序：摄像头循环 + 自动抓拍 + 实时预览
├── smile_detector.py      # 笑容检测核心模块
├── capture_test.py        # 无窗口测试：读一帧并输出分数/保存标注图
├── test_auto_capture.py   # 无窗口测试：连续读取并模拟自动抓拍
├── requirements.txt       # 依赖
├── captured/              # 抓拍照片保存目录
└── README.md
```

## 无窗口测试

如果当前环境没有显示器，可以运行无窗口测试脚本：

```bash
# 读取一帧，输出笑容分数并保存带标注的图片
python capture_test.py

# 连续读取 60 帧，模拟自动抓拍（阈值临时设为 10）
python test_auto_capture.py
```

## 单张图片测试

```bash
python smile_detector.py your_photo.jpg
```

## 调整笑容打分

笑容分数由 `smile_detector.py` 中的 `_compute_score` 方法决定，当前规则：

- 嘴角角度 0° → 0 分，3° → 100 分
- 嘴张开比例 0.40 → 0 分，0.55 → 100 分
- 最终分数 = `0.85 × 角度分 + 0.15 × 张开分`

你可以根据实际使用效果修改这些阈值和权重。调试时注意观察画面左上角的 `angle` 和 `open` 数值：

- 想让大笑更容易满分，把角度满分阈值从 3° 降到 2° 或 2.5°。
- 如果不同人自然嘴张开度差异大，可调整 `open` 的 0 分基准（当前 0.40）。
- 如果嘴张开对总分影响太小/太大，调整权重 0.85 / 0.15。

## 配置项

编辑 `config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CAMERA_INDEX` | 0 | 摄像头索引，0 为默认摄像头 |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | 1280×720 | 摄像头分辨率 |
| `SMILE_THRESHOLD` | 60 | 自动抓拍分数阈值 |
| `SMILE_CONSECUTIVE_FRAMES` | 3 | 连续满足阈值帧数，防抖 |
| `CAPTURE_COOLDOWN_FRAMES` | 30 | 两次抓拍之间的冷却帧数 |
| `SHOW_DEBUG_INFO` | True | 是否显示人脸框、关键点、分数 |
| `MIRROR_PREVIEW` | True | 预览是否镜像（自拍习惯） |

## 后续可优化方向

1. **算法升级**：用深度学习回归模型（如基于 AffectNet 训练的 MobileNet）替代几何启发式规则，准确率和稳定性更高。
2. **性能优化**：迁移到 C++ + ONNX Runtime，适配嵌入式 Linux 打卡机。
3. **活体检测**：加入眨眼、摇头等动作，防止照片/视频作弊。
4. **多目标支持**：当前只取画面中最大的人脸，可扩展为多人检测。
5. **配置界面**：用 PyQt/Tkinter 提供可视化配置窗口。
