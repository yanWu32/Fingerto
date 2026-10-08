"""检查 ML 运行环境：关键包是否安装、torch/CUDA 是否可用。

用法:
    python tools/check_env.py
（用你想验证的那个 python 解释器运行，例如 anaconda pytorch 环境）
"""
import importlib
import sys

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "gradio": "gradio",
    "numpy": "numpy",
    "pandas": "pandas",
    "openai": "openai",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "requests": "requests",
}

print("=" * 60)
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("=" * 60)

print("\n[关键包检查]")
missing = []
for mod, name in PACKAGES.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print("  [OK]  {:<16s} {}".format(name, ver))
    except Exception:
        print("  [--]  {:<16s} NOT INSTALLED".format(name))
        missing.append(name)

print("\n[torch / CUDA]")
try:
    import torch
    print("  torch:", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("  torch 不可用:", e)

print("\n[MediaPipe 可用性]")
try:
    import mediapipe as mp
    print("  mediapipe:", mp.__version__)
except Exception as e:
    print("  mediapipe 不可用:", e)

print("\n" + "=" * 60)
if missing:
    print("缺失包（需安装）:")
    for name in missing:
        print("  -", name)
else:
    print("所有关键包已就绪")
print("=" * 60)
