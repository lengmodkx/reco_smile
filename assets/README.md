# 模型文件目录

请手动下载以下模型并放置到此目录：

## 1. YuNet 人脸检测模型（必需）

下载: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

保存为: `assets/face_detection_yunet.onnx`

**此模型同时输出 5 个面部关键点**（右眼、左眼、鼻尖、嘴右角、嘴左角），用于几何特征打分。

## 2. FerPlus 表情识别模型（**已不再使用**）

~~下载: https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx~~

~~保存为: `assets/emotion-ferplus-8.onnx`~~

**注意**：从 v2 版本起已弃用 FerPlus 模型。原因是该模型对特定人脸（尤其是戴眼镜、亚洲人脸）识别效果不稳定，调试困难。新版改用 **YuNet 的 5 个关键点 + 几何特征**（嘴宽/眼距、嘴部 Y 偏移）进行打分，更可靠、更快、不需要第二个模型。

如果你已下载此模型，可以保留也可以删除。

## 验证

下载完成后，此目录应包含：
```
assets/
├── face_detection_yunet.onnx  (~340KB, 必需)
└── README.md
```
