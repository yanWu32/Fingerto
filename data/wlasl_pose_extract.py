"""WLASL（英文 ASL word-level 视频）-> 27 点骨架提取管线。

⚠️ 重要：WLASL 是英文美式手语（ASL）**视频**数据集，**不是中文主实验数据**。
本脚本仅用于「方法跨语种泛化对照基线」，产出的英文 gloss 独立保存在
`data/wlasl/glosses.txt`，**绝不混入** `data/synthetic/classes.txt`。

流程：
    视频 (User 侧已下载到 data/wlasl/videos/)
      -> MediaPipe Holistic 抽 手(21)+姿态(33) 关键点
      -> 按本项目 27 点布局映射（主手 21 + 上身 6）
      -> 截取 frame_start..frame_end
      -> padding/截断到固定 T=90
      -> normalize_sequence 归一化
      -> 存 data/wlasl/npz/{train,val,test}.npz 与 glosses.txt

27 点布局（契约锁定，见 tests/contract.py）：
    索引 0-20  : 主手 21 点（MediaPipe Hands 标准 21 点；优先左手，swap 时取右手）
    索引 21-26 : 上身 6 点（Pose 子集，索引见 POSE_LANDMARK_IDX=[0,11,12,13,14,15]
                 即 鼻/左肩/右肩/左肘/右肘/左腕）
    MediaPipe Pose 索引 15 即 LEFT_WRIST，英文 ASL 同样可用；
    若某点缺失（手未入镜 / 手腕遮挡），对应点填 0（check_skeleton_shape 仅校验有限性）。

运行（沙箱内，需要本机已把 videos 拷入）：
    conda run -n fingerto python data/wlasl_pose_extract.py \
        --meta data/wlasl/WLASL_v0.3.json \
        --videos data/wlasl/videos --out data/wlasl

合成自检（不依赖视频 / 网络 / mediapipe，仅 numpy，沙箱内可跑）：
    conda run -n fingerto python data/wlasl_pose_extract.py --self-check

数据获取见同目录 `wlasl_download_guide.md`（本 harness 网络封了
raw.githubusercontent.com 与 huggingface.co，下载须用户侧完成）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 作为脚本 `python data/wlasl_pose_extract.py` 运行时，把仓库根加入 sys.path
# 才能 import tests / perception。作为模块被 import 时不影响（已能解析）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# ---- 契约常量（与 tests/contract.py / perception/skeleton.py 对齐）----
from tests.contract import (
    NUM_POINTS,          # 27
    NUM_HAND_POINTS,     # 21
    NUM_POSE_POINTS,     # 6
    POSE_LANDMARK_IDX,   # [0,11,12,13,14,15]
    HAND_OFFSET,         # 0
    POSE_OFFSET,         # 21
    SKELETON_RANK,       # 3
    check_skeleton_shape,
)
from perception.skeleton import normalize_sequence

try:
    import cv2
    _CV_OK = True
except Exception:  # pragma: no cover
    cv2 = None
    _CV_OK = False

try:
    from perception.keypoint_extractor import (
        KeypointExtractor,
        mediapipe_available,
    )
    _MP_OK = mediapipe_available()
except Exception:  # pragma: no cover
    KeypointExtractor = None  # type: ignore
    _MP_OK = False

# 固定帧数（任务要求 T=90）
FIXED_T = 90
# 默认按 WLASL100（100 类，约 2000 条）做最小可行对照；可改 subset="full"
DEFAULT_SUBSET = "WLASL100"
# 训练/验证/测试切分比例（按样本）
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


# ---------------------------------------------------------------------------
# 1. MediaPipe 原始输出 -> 27 点映射（可独立单元测试）
# ---------------------------------------------------------------------------
def map_mediapipe_to_27(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    pose: Optional[np.ndarray],
    swap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """把单帧 MediaPipe Holistic 输出映射到 27 点骨架。

    Args:
        left_hand:  (21, 3) 或 None —— 左手 21 关键点（归一化 x,y,z）
        right_hand: (21, 3) 或 None —— 右手 21 关键点
        pose:       (33, 3) 或 None —— Pose 33 关键点
        swap:       镜像采集下 MediaPipe 的 "Left" 实为画面右侧（用户右手），
                    默认 True 时：主手优先左手，否则取右手（与项目「主手」约定一致）。

    Returns:
        skeleton: (27, 3) float32，缺失点填 0
        vis:      (27,)   float32，0/1 可见性
    """
    skeleton = np.zeros((NUM_POINTS, SKELETON_RANK), dtype=np.float32)
    vis = np.zeros((NUM_POINTS,), dtype=np.float32)

    # ---- 主手 21 点（优先左手，swap 时取右手）----
    hand = None
    if left_hand is not None:
        hand = left_hand
        if swap and right_hand is not None:
            hand = right_hand
    elif right_hand is not None:
        hand = right_hand

    if hand is not None:
        hand = np.asarray(hand, dtype=np.float32).reshape(NUM_HAND_POINTS, SKELETON_RANK)
        skeleton[HAND_OFFSET:HAND_OFFSET + NUM_HAND_POINTS] = hand
        vis[HAND_OFFSET:HAND_OFFSET + NUM_HAND_POINTS] = 1.0

    # ---- 上身 6 点（Pose 子集）----
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1, SKELETON_RANK)
        for j, idx in enumerate(POSE_LANDMARK_IDX):
            if idx < pose.shape[0]:
                skeleton[POSE_OFFSET + j] = pose[idx]
                vis[POSE_OFFSET + j] = 1.0
            else:
                # 索引越界（理论不会）：缺失点保持 0，vis=0
                vis[POSE_OFFSET + j] = 0.0

    return skeleton, vis


# ---------------------------------------------------------------------------
# 2. 固定长度 padding/截断
# ---------------------------------------------------------------------------
def fix_length(seq: np.ndarray, T: int = FIXED_T) -> np.ndarray:
    """把 (T_raw, 27, 3) 截断或尾帧重复 pad 到固定 T 帧。"""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1:] != (NUM_POINTS, SKELETON_RANK):
        raise ValueError(f"fix_length 期望 (T,27,3)，收到 {seq.shape}")
    n = seq.shape[0]
    if n == T:
        return seq
    if n > T:
        return seq[:T]
    # 尾帧重复 pad（保持末态静止，对识别更友好）
    pad = np.broadcast_to(seq[-1:], (T - n, NUM_POINTS, SKELETON_RANK))
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 单视频提取（真实路径，需 cv2 + MediaPipe）
# ---------------------------------------------------------------------------
def extract_one_video(
    video_path: str,
    extractor,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    T: int = FIXED_T,
) -> Optional[np.ndarray]:
    """读视频 -> 逐帧 27 点 -> 截取 frame_start..frame_end -> pad/截断 -> 归一化。

    Returns:
        (T, 27, 3) float32；视频无法打开 / 无帧时返回 None。
    """
    if not _CV_OK:
        raise RuntimeError("未安装 opencv-python，无法读帧；请 pip install opencv-contrib-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # 区间语义：frame_start=None 或 <0 -> 从头；frame_end=None 或 <=0 -> 到尾
    # （WLASL 标注里 frame_end=-1 表示「整段视频」，不是「第 -1 帧」）
    f_start = frame_start if (frame_start is not None and frame_start > 0) else 0
    f_end = frame_end if (frame_end is not None and frame_end > 0) else (1 << 30)

    frames: List[np.ndarray] = []
    vis_frames: List[np.ndarray] = []
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 仅保留 [f_start, f_end] 区间
        if fidx < f_start:
            fidx += 1
            continue
        if fidx > f_end:
            break
        skel, vis = extractor.extract_frame(frame)
        frames.append(skel)
        vis_frames.append(vis)
        fidx += 1
    cap.release()

    if not frames:
        return None

    seq = np.stack(frames, axis=0).astype(np.float32)
    seq = fix_length(seq, T=T)
    # 归一化（鼻原点 / 肩宽尺度 / 肩线水平）
    norm, _, _ = normalize_sequence(seq)
    norm = fix_length(norm, T=T)  # 归一化可能删帧，重新固定长度
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. 数据集构建（读 WLASL_v0.3.json -> 切分 -> 存 npz + glosses.txt）
# ---------------------------------------------------------------------------
def _normalize_meta(meta) -> dict:
    """把 WLASL 注释统一成 {gloss: [instances]} 字典。

    WLASL_v0.3.json / WLASL100.json 实际是 list[{gloss, instances:[...]}]；
    旧代码假设 dict[gloss] = list。这里做兼容归一，避免 AttributeError: 'list'。
    """
    if isinstance(meta, list):
        d = {}
        for item in meta:
            g = item["gloss"]
            d.setdefault(g, [])
            d[g].extend(item.get("instances", []))
        return d
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"不支持的 meta 类型：{type(meta)}")


def _select_subset(meta: dict, subset: str) -> dict:
    """按 subset 过滤 gloss 词：WLASL100/300/1000/full。"""
    if subset in ("full", "all"):
        return meta
    # subset 形如 WLASL100 -> 取样本数前 100 的 gloss
    if subset.lower().startswith("wlasl"):
        try:
            k = int(subset.lower().replace("wlasl", ""))
        except ValueError:
            return meta
        ranked = sorted(meta.items(), key=lambda kv: len(kv[1]), reverse=True)
        return {g: s for g, s in ranked[:k]}
    return meta


def _split_samples(entries: List[dict], ratios: Dict[str, float], seed: int = 20261010):
    """把样本按 gloss 内分层切分到 train/val/test。"""
    rng = np.random.default_rng(seed)
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for e in entries:
        r = float(rng.random())
        if r < ratios["train"]:
            splits["train"].append(e)
        elif r < ratios["train"] + ratios["val"]:
            splits["val"].append(e)
        else:
            splits["test"].append(e)
    return splits


def build_dataset(
    meta_path: str,
    videos_dir: str,
    out_dir: str,
    subset: str = DEFAULT_SUBSET,
    T: int = FIXED_T,
    ratios: Dict[str, float] = SPLIT_RATIOS,
    skip_missing: bool = True,
) -> dict:
    """构建 WLASL 27 点骨架数据集并落盘。

    Args:
        meta_path:    WLASL_v0.3.json 路径
        videos_dir:   已下载视频目录（含 <video_id>.mp4）
        out_dir:      输出目录（将写入 npz/{train,val,test}.npz 与 glosses.txt）
        subset:       "WLASL100" | "WLASL300" | "WLASL1000" | "full"
        skip_missing: 视频缺失/抽取失败时跳过并记录（不阻塞整批）

    Returns:
        info: dict，含各切分样本数 / 类数 / 跳过清单
    """
    if not _MP_OK:
        raise RuntimeError("MediaPipe 不可用，无法抽取骨架；请先 pip install mediapipe==0.10.14")
    meta_path = Path(meta_path)
    videos_dir = Path(videos_dir)
    out_dir = Path(out_dir)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"未找到 {meta_path}。请先在个人机器按 wlasl_download_guide.md 下载"
            " WLASL_v0.3.json 并拷到 data/wlasl/。"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = _normalize_meta(meta)          # list -> dict（兼容 WLASL_v0.3.json 新格式）
    meta = _select_subset(meta, subset)

    # 展开为 (video_id, gloss, frame_start, frame_end, url)
    samples: List[dict] = []
    for gloss, entries in meta.items():
        for e in entries:
            samples.append({
                "video_id": e["video_id"],
                "gloss": gloss,
                "frame_start": int(e.get("frame_start", 0)),
                "frame_end": int(e.get("frame_end", 1 << 30)),
                "url": e.get("url", ""),
            })

    classes = sorted({s["gloss"] for s in samples})
    class_to_idx = {g: i for i, g in enumerate(classes)}

    splits = _split_samples(samples, ratios)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)

    info: dict = {"subset": subset, "num_classes": len(classes), "splits": {}}
    skipped: List[str] = []

    with KeypointExtractor() as ext:  # type: ignore[arg-type]
        for split, split_samples in splits.items():
            skeletons, labels = [], []
            for s in split_samples:
                vid_path = videos_dir / f"{s['video_id']}.mp4"
                # 兼容其它扩展名
                if not vid_path.exists():
                    cands = list(videos_dir.glob(f"{s['video_id']}.*"))
                    vid_path = cands[0] if cands else vid_path
                if not vid_path.exists():
                    if skip_missing:
                        skipped.append(s["video_id"])
                        continue
                    else:
                        raise FileNotFoundError(f"缺失视频：{vid_path}")
                try:
                    seq = extract_one_video(
                        str(vid_path), ext,
                        frame_start=s["frame_start"],
                        frame_end=s["frame_end"],
                        T=T,
                    )
                except Exception as e:  # 单样本失败不阻塞
                    if skip_missing:
                        skipped.append(f"{s['video_id']}:{e}")
                        continue
                    raise
                if seq is None:
                    if skip_missing:
                        skipped.append(s["video_id"])
                        continue
                    raise RuntimeError(f"视频无有效帧：{vid_path}")
                if not check_skeleton_shape(seq):
                    if skip_missing:
                        skipped.append(f"{s['video_id']}:contract_fail")
                        continue
                    raise RuntimeError(f"样本未通过 check_skeleton_shape：{vid_path}")
                skeletons.append(seq)
                labels.append(class_to_idx[s["gloss"]])

            if not skeletons:
                info["splits"][split] = {"num_samples": 0, "shape": None}
                continue
            np.savez(
                out_dir / "npz" / f"{split}.npz",
                skeletons=np.stack(skeletons, axis=0).astype(np.float32),
                labels=np.array(labels, dtype=np.int64),
                classes=np.array(classes, dtype=object),
            )
            info["splits"][split] = {
                "num_samples": len(skeletons),
                "shape": (len(skeletons), T, NUM_POINTS, SKELETON_RANK),
            }

    # 英文 gloss 列表独立保存（不混入中文 classes.txt）
    (out_dir / "glosses.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )
    info["skipped"] = skipped
    info["glosses_file"] = str(out_dir / "glosses.txt")
    print(f"[wlasl] subset={subset} 类数={len(classes)} "
          f"各切分={ {k: v['num_samples'] for k, v in info['splits'].items()} } "
          f"跳过={len(skipped)}")
    return info


# ---------------------------------------------------------------------------
# 5. 合成自检（不依赖视频/网络/mediapipe，仅 numpy；沙箱可跑）
# ---------------------------------------------------------------------------
def _synthetic_self_check(
    T: int = FIXED_T,
    raw_frames: int = 120,
    rng_seed: int = 20261010,
) -> dict:
    """用随机张量模拟 MediaPipe Holistic 输出，走完整映射->截取->裁剪->归一化，
    断言输出满足 (T,27,3) 契约且 check_skeleton_shape 为真。

    用于无视频/无网络环境下验证管线逻辑正确（与真实路径复用同一 map/fix_length）。
    """
    rng = np.random.default_rng(rng_seed)

    # 模拟逐帧 MediaPipe 原始输出：左手 21、右手 21、Pose 33，均为有限随机值
    raw_seq: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(raw_frames):
        # 偶尔模拟手未入镜（None）或手腕遮挡（pose 点置 0）
        lh = rng.normal(0.5, 0.1, size=(NUM_HAND_POINTS, 3)).astype(np.float32)
        rh = rng.normal(0.5, 0.1, size=(NUM_HAND_POINTS, 3)).astype(np.float32)
        pose = rng.normal(0.5, 0.1, size=(33, 3)).astype(np.float32)
        if rng.random() < 0.1:
            lh = None  # 左手偶尔缺失
        pose[15] = 0.0  # 模拟左腕遮挡（缺失点填 0）
        raw_seq.append((lh, rh, pose))

    # 映射：复用与真实路径相同的 map_mediapipe_to_27
    mapped = []
    for lh, rh, pose in raw_seq:
        skel, _ = map_mediapipe_to_27(lh, rh, pose, swap=True)
        assert skel.shape == (NUM_POINTS, SKELETON_RANK)
        assert np.all(np.isfinite(skel)), "映射结果必须有限"
        mapped.append(skel)
    seq = np.stack(mapped, axis=0).astype(np.float32)  # (raw_frames, 27, 3)

    # 模拟 meta 的 frame_start..frame_end 截取
    frame_start, frame_end = 10, 80
    crop = seq[frame_start:frame_end + 1]
    cropped_len = crop.shape[0]
    assert cropped_len == (frame_end - frame_start + 1), f"截取长度异常：{cropped_len}"

    # 固定长度到 T=90（裁剪区间 < T -> 尾帧 pad）
    fixed = fix_length(crop, T=T)
    assert fixed.shape == (T, NUM_POINTS, SKELETON_RANK), f"fix_length 形状异常：{fixed.shape}"

    # 归一化（与真实路径一致）
    norm, _, _ = normalize_sequence(fixed)
    norm = fix_length(norm, T=T)

    # 契约校验
    assert norm.shape == (T, NUM_POINTS, SKELETON_RANK), f"最终形状异常：{norm.shape}"
    assert norm.dtype == np.float32
    assert np.all(np.isfinite(norm)), "归一化后存在 NaN/Inf"
    assert check_skeleton_shape(norm), "check_skeleton_shape 应为真"

    # 映射正确性抽样断言：右手点 k 的值应原样落在 skeleton[k]
    lh, rh, pose = raw_seq[50]
    skel, _ = map_mediapipe_to_27(lh, rh, pose, swap=True)
    if rh is not None:  # swap=True 时主手取右手
        assert np.allclose(skel[5], rh[5]), "右手食指根未正确映射"
    assert np.allclose(skel[26], pose[15]), "Pose 左腕(15) 未映射到索引 26"

    return {
        "raw_frames": raw_frames,
        "cropped_len": cropped_len,
        "final_shape": list(norm.shape),
        "contract_ok": bool(check_skeleton_shape(norm)),
        "all_finite": bool(np.all(np.isfinite(norm))),
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="WLASL -> 27 点骨架提取 / 合成自检")
    ap.add_argument("--meta", default="data/wlasl/WLASL_v0.3.json")
    ap.add_argument("--videos", default="data/wlasl/videos")
    ap.add_argument("--out", default="data/wlasl")
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--T", type=int, default=FIXED_T)
    ap.add_argument("--self-check", action="store_true",
                    help="运行合成自检（无需视频/网络/mediapipe）")
    args = ap.parse_args()

    if args.self_check:
        res = _synthetic_self_check(T=args.T)
        print("[self-check] 合成自检通过：", res)
        return

    info = build_dataset(
        meta_path=args.meta,
        videos_dir=args.videos,
        out_dir=args.out,
        subset=args.subset,
        T=args.T,
    )
    print("[wlasl] 完成：", json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
