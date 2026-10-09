"""感知层 — OpenCV 采集 + MediaPipe 关键点提取 + 归一化。

输出：27 点骨架序列 (T, 27, 3)
  - 双手各 21 点（MediaPipe Hands）
  - 上半身 6 点（Pose 子集：鼻/双肩/双肘/左腕）
"""
