# WLASL 数据集获取指南（用户侧操作）

> **为什么这份指南必须用户自己执行？**
> 本 DeepSeek Harness 沙箱网络封禁了 `raw.githubusercontent.com` 与 `huggingface.co`
> （解析到非公网 IP），因此**元数据下载与视频下载都无法在沙箱内完成**。
> 下面的步骤需要你在本机（有外网、有浏览器/代理的机器）上执行，
> 然后把产物（JSON 元数据 + 视频目录）拷回本仓库的 `data/wlasl/` 目录即可。

---

## 一、取元数据 `WLASL_v0.3.json`

WLASL 是 **word-level 美式手语（ASL）视频数据集**，本身**不含骨架**，
只提供「YouTube 视频 URL + 起止帧 + gloss 词」的标注。
元数据来自官方仓库 <https://github.com/dxli94/WLASL>，主文件即 `WLASL_v0.3.json`。

### 方法 A（推荐）：本机用 git 或浏览器下载

```bash
# 方式 1：git 克隆（含 WLASL_v0.3.json）
git clone https://github.com/dxli94/WLASL.git
# 然后复制其中的 WLASL_v0.3.json 到仓库：
cp WLASL/WLASL_v0.3.json <本仓库>/Fingerto/data/wlasl/WLASL_v0.3.json

# 方式 2：直接用浏览器打开下面的 raw 链接下载 JSON
# https://raw.githubusercontent.com/dxli94/WLASL/master/WLASL_v0.3.json
```

> 注意：`raw.githubusercontent.com` 在本沙箱被封，但**在你本机是正常可访问的**，
> 所以请用浏览器/本机网络下载，再拷回沙箱目录。

### 元数据字段说明

`WLASL_v0.3.json` 大致结构（按 gloss 词聚合）：

```json
{
  "wish": [
    {"video_id": "2_BwcV1Xn8", "frame_start": 2,  "frame_end": 42, "url": "https://www.youtube.com/watch?v=2_BwcV1Xn8"},
    {"video_id": "6aM...",      "frame_start": 10, "frame_end": 60, "url": "..."}
  ],
  "hello": [ ... ],
  ...
}
```

- `video_id`：YouTube 视频 ID（即 `watch?v=<video_id>`）。
- `frame_start` / `frame_end`：**关键帧区间**（基于该视频原始 fps，通常 29.97）。
- `url`：YouTube 播放页。

---

## 二、用 yt-dlp 批量下载视频到 `data/wlasl/videos/`

### 1. 安装依赖（本机）

```bash
pip install yt-dlp
# 视频解码还需要 ffmpeg，官网下载并加入 PATH：
# https://ffmpeg.org/download.html
```

### 2. 下载（推荐先做 WLASL100）

```bash
cd <本仓库>/Fingerto/data/wlasl

# 先把所有 video_id 导出成下载清单（一行一个 id）
python - <<'PY'
import json
meta = json.load(open("WLASL_v0.3.json"))
ids = set()
for samples in meta.values():
    for s in samples:
        ids.add(s["video_id"])
with open("video_ids.txt", "w", encoding="utf-8") as f:
    for vid in sorted(ids):
        f.write(vid + "\n")
print("共", len(ids), "个视频待下载")
PY

# 用 yt-dlp 批量下载（mp4，720p 够用；失败自动跳过）
yt-dlp -a video_ids.txt \
  -o "videos/%(id)s.%(ext)s" \
  -f "mp4/best[height<=720]" \
  --merge-output-format mp4 \
  --ignore-errors --no-warnings
```

> `-a video_ids.txt`：从清单逐行读 video_id（yt-dlp 会自动拼成 YouTube URL）。
> `--ignore-errors`：某条视频下架/区域限制时跳过，**不阻塞整批**（与 TASK-007 注意点 4 一致）。

### 3. 把产物拷回本仓库（`data/wlasl/`）

```
data/wlasl/
├── WLASL_v0.3.json        # 元数据
├── video_ids.txt          # 下载清单（可选）
└── videos/                # 下载好的 mp4（*.npz 不入库，已 .gitignore）
```

此时沙箱里的 `data/wlasl_pose_extract.py` 即可读取这些视频跑骨架提取。

---

## 三、与中文数据集/gloss 语种的差异（务必注意）

| 维度 | 中文数据集（CSL / 主实验） | WLASL（英文 ASL 对照） |
| --- | --- | --- |
| 语种 | 中文手语词（如「你/我/好」） | 英文美式手语单词（如 `wish/hello`） |
| 用途 | **主实验训练/测试集** | **仅作跨语种方法泛化对照基线** |
| gloss 列表 | `data/synthetic/classes.txt`（中文 60 类） | `data/wlasl/glosses.txt`（英文 ASL 词，独立保存） |
| 是否入库骨架 | 是 | npz 不入库（`*.npz` 已被忽略），仅入库代码+gloss 文本 |

⚠️ **不要把 WLASL 的英文 gloss 混入中文 `classes.txt`，也不要用 WLASL 当中文主实验结果。**
它只在论文里作为「方法在跨语种（英文 ASL）数据上的泛化性」补充证据。

---

## 四、常见问题

- **YouTube 区域限制/下架**：`--ignore-errors` 跳过即可，缺失样本记录在 README 的「跳过清单」。
- **只有视频没有骨架**：正常。骨架由 `data/wlasl_pose_extract.py` 在沙箱/本机用 MediaPipe 生成。
- **frame 区间越界**：提取脚本会 `min(frame_end, 总帧数-1)` 保护，详见 `wlasl_pose_extract.py`。
