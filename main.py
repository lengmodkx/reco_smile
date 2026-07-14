"""
笑容识别拍照程序入口。

运行方式：
    python main.py

按键：
    q / ESC : 退出
    空格    : 手动抓拍当前画面
"""
import os
import time
from datetime import datetime

import cv2

import config
from smile_detector import SmileDetector, SmileResult


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def draw_debug_info(frame, result: SmileResult) -> None:
    """在画面上绘制调试信息：人脸框、关键点、分数。"""
    x, y, w, h = result.face_box

    # 人脸框，颜色根据分数渐变
    color = (
        int(255 * (1 - result.score / 100)),
        int(255 * (result.score / 100)),
        0,
    )
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # 关键点（只画嘴唇附近用于观察，全部点会太密）
    lip_indices = [SmileDetector.LEFT_CORNER, SmileDetector.RIGHT_CORNER,
                   SmileDetector.UPPER_LIP, SmileDetector.LOWER_LIP]
    for idx in lip_indices:
        px, py = result.landmarks[idx]
        cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

    # 嘴角连线
    lc = result.landmarks[SmileDetector.LEFT_CORNER]
    rc = result.landmarks[SmileDetector.RIGHT_CORNER]
    cv2.line(frame, lc, rc, (255, 255, 255), 1)

    # 文字信息
    text = f"Smile: {result.score:.0f}"
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    detail = f"angle={result.angle:.1f} open={result.mouth_open_ratio:.2f}"
    cv2.putText(frame, detail, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def save_photo(frame, result: SmileResult) -> str:
    """保存照片到 captured 目录，返回保存路径。"""
    ensure_dir(config.CAPTURE_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"smile_{timestamp}_score{int(result.score)}.jpg"
    filepath = os.path.join(config.CAPTURE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def main() -> None:
    ensure_dir(config.CAPTURE_DIR)
    detector = SmileDetector()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"无法打开摄像头，索引: {config.CAMERA_INDEX}")
        print("提示：可尝试修改 config.py 中的 CAMERA_INDEX（0/1/2...）")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    consecutive_count = 0
    cooldown = 0
    print("程序已启动，按 'q' 退出，按 '空格' 手动拍照。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头失败")
            break

        # 镜像显示（自拍习惯）
        display_frame = cv2.flip(frame, 1) if config.MIRROR_PREVIEW else frame

        result = detector.detect(display_frame)
        captured_path = None

        if result:
            if config.SHOW_DEBUG_INFO:
                draw_debug_info(display_frame, result)

            # 自动拍照逻辑
            if result.score >= config.SMILE_THRESHOLD:
                consecutive_count += 1
                if cooldown <= 0 and consecutive_count >= config.SMILE_CONSECUTIVE_FRAMES:
                    captured_path = save_photo(frame, result)  # 保存原始帧，不保存镜像
                    print(f"[自动抓拍] 分数: {result.score:.0f} -> {captured_path}")
                    consecutive_count = 0
                    cooldown = config.CAPTURE_COOLDOWN_FRAMES
            else:
                consecutive_count = 0

        if cooldown > 0:
            cooldown -= 1

        # 提示文字
        status = "Detecting..." if result else "No face"
        cv2.putText(display_frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smile Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q 或 ESC
            break
        elif key == ord(" ") and result:
            captured_path = save_photo(frame, result)
            print(f"[手动抓拍] 分数: {result.score:.0f} -> {captured_path}")

    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == "__main__":
    main()
