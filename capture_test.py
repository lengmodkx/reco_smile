"""无窗口测试：读取一帧摄像头画面并检测笑容，保存结果图。"""
import os
import cv2
from smile_detector import SmileDetector
from main import draw_debug_info


def main():
    detector = SmileDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    ret, frame = cap.read()
    if not ret:
        print("读取画面失败")
        cap.release()
        return

    # 镜像一下，和主程序保持一致
    frame = cv2.flip(frame, 1)
    result = detector.detect(frame)

    if result is None:
        print("未检测到人脸，已保存原始画面到 test_frame.jpg")
        cv2.imwrite("test_frame.jpg", frame)
    else:
        print(f"检测到人脸，笑容分数: {result.score}")
        print(f"嘴角角度: {result.angle:.2f}°，嘴张开比: {result.mouth_open_ratio:.3f}")
        draw_debug_info(frame, result)
        cv2.imwrite("test_frame.jpg", frame)
        print("已保存带标注的画面到 test_frame.jpg")

    detector.release()
    cap.release()


if __name__ == "__main__":
    main()
