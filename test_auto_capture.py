"""无窗口自动抓拍测试：连续读取若干帧，分数达标即保存。"""
import cv2
from smile_detector import SmileDetector
from main import save_photo


def main():
    threshold = 10  # 测试用低阈值
    detector = SmileDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    saved = []
    for i in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        display = cv2.flip(frame, 1)
        result = detector.detect(display)
        if result and result.score >= threshold:
            path = save_photo(frame, result)
            saved.append((i, result.score, path))
            print(f"frame {i}: score={result.score}, saved={path}")

    cap.release()
    detector.release()
    print(f"\n共抓拍 {len(saved)} 张")


if __name__ == "__main__":
    main()
