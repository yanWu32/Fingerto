"""项目结构体检 — 检查工程骨架是否完整、可导入。

用法:
    python tools/check_project.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_DIRS = [
    "perception", "segmentation", "recognition", "recognition/models",
    "agent", "app", "data", "experiments", "configs", "docs", "tools",
]
REQUIRED_FILES = [
    "README.md", "requirements.txt", ".gitignore",
    "configs/perception.yaml", "configs/segmentation.yaml",
    "configs/recognition.yaml", "configs/agent.yaml",
    "configs/experiment.yaml", "configs/__init__.py",
]
PACKAGES = ["configs", "perception", "segmentation", "recognition", "agent", "app", "experiments"]

ok, fail, warn = [], [], []


def check_dirs():
    for d in REQUIRED_DIRS:
        p = ROOT / d
        (ok if p.is_dir() else fail).append(f"目录 {d}")


def check_files():
    for f in REQUIRED_FILES:
        p = ROOT / f
        (ok if p.is_file() else fail).append(f"文件 {f}")


def check_imports():
    for pkg in PACKAGES:
        try:
            __import__(pkg)
            ok.append(f"导入 {pkg}")
        except Exception as e:
            fail.append(f"导入 {pkg} -> {type(e).__name__}: {e}")


def check_configs():
    try:
        from configs import list_configs, load_config
        names = list_configs()
        if not names:
            fail.append("configs 下无 yaml")
            return
        for n in names:
            cfg = load_config(n)
            ok.append(f"配置 {n}.yaml ({len(cfg.to_dict())} 个顶层键)")
    except Exception as e:
        fail.append(f"配置加载 -> {type(e).__name__}: {e}")


def check_data_gitkeep():
    for sub in ["data", "recognition/models"]:
        gk = ROOT / sub / ".gitkeep"
        if gk.exists():
            ok.append(f"{sub}/.gitkeep 存在")
        else:
            warn.append(f"{sub}/.gitkeep 缺失（空目录不会被 Git 跟踪）")


def check_env_file():
    env = ROOT / ".env"
    if env.exists():
        ok.append(".env 存在（LLM API Key 已配置）")
    else:
        warn.append(".env 不存在 — 智能体层需配置 API Key（参考 docs/环境配置指南.md 步骤7）")


def main():
    print("=" * 60)
    print("Fingerto 项目结构体检")
    print("=" * 60)
    print(f"项目根: {ROOT}\n")

    check_dirs()
    check_files()
    check_imports()
    check_configs()
    check_data_gitkeep()
    check_env_file()

    print(f"[通过] {len(ok)} 项")
    for x in ok:
        print(f"  [OK]   {x}")

    if warn:
        print(f"\n[提示] {len(warn)} 项")
        for x in warn:
            print(f"  [--]   {x}")

    if fail:
        print(f"\n[失败] {len(fail)} 项")
        for x in fail:
            print(f"  [FAIL] {x}")

    print("\n" + "=" * 60)
    if fail:
        print("结果: 存在问题，请修复上述 [FAIL] 项")
        return 1
    print("结果: 工程骨架完整 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
