"""骨架图工具：从 perception/skeleton.py 的 HAND_EDGES / POSE_EDGES 构造 27 点邻接。

布局（唯一事实源，见 tests/contract.py 与 perception/skeleton.py）：
    - 主手 21 点：HAND_OFFSET = 0  （MediaPipe Hands 标准 21 点）
    - 上身 6 点 ：POSE_OFFSET = 21 （鼻/左肩/右肩/左肘/右肘/左腕）
图 = HAND_EDGES（主手内部骨骼，20 条边）+ POSE_EDGES（上身骨骼，7 条边）。
注意：单主手 + 上身，不假设双手。
"""

from __future__ import annotations

import numpy as np

try:  # 共享骨架定义，失败则回退到内联常量（避免阻断）
    from perception.skeleton import HAND_EDGES, POSE_EDGES
except Exception:  # pragma: no cover - 仅在骨架模块缺失时
    HAND_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    # 上身 6 点相对索引（0-5），接入 27 点布局时须 +POSE_OFFSET
    POSE_EDGES = [
        (0, 1), (0, 2), (1, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 5),
    ]

NUM_POINTS = 27
NUM_HAND_POINTS = 21
NUM_POSE_POINTS = 6
HAND_OFFSET = 0
POSE_OFFSET = 21

# ⚠️ 关键点：perception/skeleton.py 里 POSE_EDGES 用「上身相对索引 0-5」定义
# （与 perception/visualize.py 的 `a + POSE_OFFSET` 用法一致）。接入 27 点全局
# 布局时必须 +POSE_OFFSET，否则会与主手 0-5 点碰撞造出错误手内边，而上身
# 21-26 在图中彻底孤立 —— ST-GCN 向心分区(BFS 从主手腕 0 出发)将够不到上身点，
# 把它们全赋成默认标签 center。故：
#   1) POSE_EDGES 整体 +POSE_OFFSET 平移到 21-26 内部边；
#   2) 补一条 HAND→POSE 跨部件连接边（主手腕 0 ↔ 上身左腕 26），使整图连通。
POSE_EDGES_GLOBAL = [(i + POSE_OFFSET, j + POSE_OFFSET) for i, j in POSE_EDGES]
HAND_POSE_BRIDGE = [(HAND_OFFSET, POSE_OFFSET + 5)]  # 主手腕(0) ↔ 上身左腕(26)

# 单主手 + 上身：图邻接 = 主手边 + 上身边(已偏移) + 跨部件桥接边（索引对齐 0..26）
SKELETON_EDGES: list[tuple[int, int]] = (
    list(HAND_EDGES) + POSE_EDGES_GLOBAL + HAND_POSE_BRIDGE
)


def assert_graph_connected(num_points: int = NUM_POINTS) -> None:
    """断言 27 点骨架图连通（所有节点均可从根点 0 BFS 到达）。

    用于回归测试：一旦 POSE_EDGES 偏移或桥接边被误删，这里会立即失败。
    """
    adj = build_adjacency(num_points, self_loop=False)
    seen = {0}
    q = [0]
    while q:
        u = q.pop()
        for v in range(num_points):
            if adj[u, v] > 0 and v not in seen:
                seen.add(v)
                q.append(v)
    missing = [v for v in range(num_points) if v not in seen]
    if missing:
        raise AssertionError(f"骨架图存在孤立点（未连通）：{missing}")


def build_adjacency(num_points: int = NUM_POINTS, self_loop: bool = True) -> np.ndarray:
    """构造无向骨架邻接矩阵 (N, N)，对角线自环。

    边来自 SKELETON_EDGES（HAND_EDGES + POSE_EDGES），索引已对齐 27 点布局。
    """
    A = np.zeros((num_points, num_points), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        if i < num_points and j < num_points:
            A[i, j] = 1.0
            A[j, i] = 1.0
    if self_loop:
        for i in range(num_points):
            A[i, i] = 1.0
    return A


def build_edge_index(num_points: int = NUM_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """返回 COO 形式边索引 (2, E)。"""
    src, dst = [], []
    for i, j in SKELETON_EDGES:
        if i < num_points and j < num_points:
            src.append(i); dst.append(j)
            src.append(j); dst.append(i)
    if not src:  # 退化保护
        src, dst = [0], [0]
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


# ---- ST-GCN 分区策略（单一主手+上身，按向心距离分区）----
def partition_strategy(num_points: int = NUM_POINTS) -> np.ndarray:
    """向心分区：以根点（主手腕 0）为 center，按到根的最短跳数赋 0/1/2。

    返回 (N,) 的整数标签：0=center, 1=向心（靠近根）, 2=离心（远离根）。
    """
    import collections

    # BFS 从根点 0
    adj = build_adjacency(num_points, self_loop=False)
    dist = {0: 0}
    q = collections.deque([0])
    while q:
        u = q.popleft()
        for v in range(num_points):
            if adj[u, v] > 0 and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    labels = np.zeros(num_points, dtype=np.int64)
    for v, d in dist.items():
        labels[v] = 0 if d == 0 else (1 if d <= 2 else 2)
    return labels
