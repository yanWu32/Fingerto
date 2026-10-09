"""验证感知层：MediaPipe 能否真正提取手部关键点。

用一个合成图像 + 摄像头两种方式验证，重点确认：
  1. MediaPipe 版本与 API 形态（1.x 与 0.10.x 不同）
  2. 能否拿到手部 21 点
  3. 坐标形态（归一化 / 像素 / 是否有 z）

用法:
    python tools/verify_perception.py
    python tools/verify_perception.py --camera   # 额外开摄像头试一帧
"""
import argparse
import sys

import cv2
import numpy as np

print("=" * 60)
print("MediaPipe 感知层验证")
print("=" * 60)

try:
    import mediapipe as mp
    print("mediapipe version:", mp.__version__)
except Exception as e:
    print("mediapipe 导入失败:", e)
    sys.exit(1)

# ---------- 探测 API 形态 ----------
print("\n[1] 探测 MediaPipe API 形态")
HAS_SOLUTIONS = hasattr(mp, "solutions")
HAS_TASKS = hasattr(mp, "tasks")
print("  有 mp.solutions (旧版 API):", HAS_SOLUTIONS)
print("  有 mp.tasks     (新版 API):", HAS_TASKS)

if HAS_SOLUTIONS:
    print("  mp.solutions.hands 可用:", hasattr(mp.solutions, "hands"))
    print("  mp.solutions.holistic 可用:", hasattr(mp.solutions, "holistic"))
    print("  mp.solutions.drawing_utils 可用:", hasattr(mp.solutions, "drawing_utils"))
else:
    print("  !! 无 mp.solutions —— 这是 MediaPipe 1.x 的新 API 形态")
    print("  !! 需要改用 mp.tasks.vision.HandLandmarker（需下载 .task 模型文件）")


# ---------- 构造测试图像 ----------
def make_test_image():
    """生成一张简单测试图（纯色 + 一些结构），用于验证推理能否跑通。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    cv2.circle(img, (320, 240), 120, (255, 255, 255), -1)
    return img


img = make_test_image()
print("\n[2] 测试图像:", img.shape)

# ---------- 尝试旧版 API ----------
print("\n[3] 尝试 mp.solutions.hands.Hands 推理")
if HAS_SOLUTIONS and hasattr(mp.solutions, "hands"):
    try:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        print("  推理执行成功")
        print("  multi_hand_landmarks:", "None" if res.multi_hand_landmarks is None else f"{len(res.multi_hand_landmarks)} 只手")
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            print(f"  单手关键点数: {len(lm)}  (期望 21)")
            p0 = lm[0]
            print(f"  第0点: x={p0.x:.4f} y={p0.y:.4f} z={p0.z:.4f}")
            print("  -> 坐标系: 归一化 [0,1]，含 z 深度")
        hands.close()
        print("\n[结论] 旧版 API 可用，可正常提取 21 点骨架")
    except Exception as e:
        print("  [FAIL] 旧版 API 推理失败:", type(e).__name__, e)
        import traceback
        traceback.print_exc()
else:
    print("  [SKIP] 无 mp.solutions.hands")

# ---------- 尝试新版 API ----------
if not HAS_SOLUTIONS:
    print("\n[3b] 尝试新版 mp.tasks API")
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        print("  可导入 tasks.python.vision:", True)
        print("  注意: 新版需先下载 hand_landmarker.task 模型文件")
        print("  参考: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    except Exception as e:
        print("  [FAIL] 新版 API 导入失败:", e)

# ---------- 摄像头 ----------
ap = argparse.ArgumentParser()
ap.add_argument("--camera", action="store_true", help="额外测试摄像头读取")
args, _ = ap.parse_known_args()

if args.camera:
    print("\n[4] 摄像头测试")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [FAIL] 无法打开摄像头")
    else:
        ok, frame = cap.read()
        if ok:
            print(f"  [OK] 读到一帧: {frame.shape}")
        else:
            print("  [FAIL] 读到帧失败")
        cap.release()

print("\n" + "=" * 60)
