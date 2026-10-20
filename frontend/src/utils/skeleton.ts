// 27 点骨架模型与可视化常量。
// 事实源：Fingerto/perception/skeleton.py（NUM_POINTS / HAND_EDGES / POSE_EDGES）
// 前端演示用：把识别层的输入张量 (T, 27, 3) 渲染成可播放的骨架动画。
// 坐标含义：x 向右、y 向下（屏幕系）、z 深度；归一化空间以鼻为原点、肩宽为尺度。

/** 骨架总点数（契约 NUM_POINTS） */
export const NUM_POINTS = 27
/** 主手点数（MediaPipe Hands 标准 21 点） */
export const NUM_HAND_POINTS = 21
/** 上半身点数（Pose 子集 6 点） */
export const NUM_POSE_POINTS = 6
export const HAND_OFFSET = 0
export const POSE_OFFSET = 21

/** 单个点 [x, y, z] */
export type Point3 = [number, number, number]
/** 单帧骨架：27 个点 */
export type SkeletonFrame = Point3[]

/** 主手骨连接（索引 0-20，来自 perception/skeleton.py HAND_EDGES） */
export const HAND_EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
]

/** 上身骨连接（相对索引，来自 POSE_EDGES；+POSE_OFFSET 后为绝对索引） */
export const POSE_EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 2], // 鼻-双肩三角
  [1, 3],
  [3, 5], // 左肩-左肘-左腕
  [2, 4],
  [4, 5], // 右肩-右肘-左腕
]

/** 上身骨连接（绝对索引） */
export const POSE_ABS_EDGES: [number, number][] = POSE_EDGES.map(
  ([a, b]) => [a + POSE_OFFSET, b + POSE_OFFSET] as [number, number]
)

/** 主手 21 点名称（0 手腕, 1-4 拇指, 5-8 食指, 9-12 中指, 13-16 无名指, 17-20 小指） */
export const HAND_POINT_NAMES = [
  '手腕',
  '拇指CMC',
  '拇指MCP',
  '拇指IP',
  '拇指尖',
  '食指MCP',
  '食指PIP',
  '食指DIP',
  '食指尖',
  '中指MCP',
  '中指PIP',
  '中指DIP',
  '中指尖',
  '无名指MCP',
  '无名指PIP',
  '无名指DIP',
  '无名指尖',
  '小指MCP',
  '小指PIP',
  '小指DIP',
  '小指尖',
]

/** 上半身 6 点名称（鼻 / 左肩 / 右肩 / 左肘 / 右肘 / 左腕） */
export const POSE_POINT_NAMES = ['鼻', '左肩', '右肩', '左肘', '右肘', '左腕']

/** 取第 i 个点的名称（自动区分手/上身） */
export function pointName(i: number): string {
  return i < NUM_HAND_POINTS ? HAND_POINT_NAMES[i] : POSE_POINT_NAMES[i - POSE_OFFSET]
}

// ---------------------------------------------------------------------------
// 参数化手模型：由「手掌中心 + 旋转角 + 五指弯曲度」生成 21 个关键点
// ---------------------------------------------------------------------------

/** 五指弯曲度 [拇指, 食指, 中指, 无名指, 小指]，取值 0（伸直）~1（完全握拳） */
export type FiveCurl = [number, number, number, number, number]

/** 单手姿态参数 */
export interface HandPose {
  /** 手腕（即第 0 点）位置，归一化坐标 */
  wrist: [number, number]
  /** 手掌整体旋转角（弧度，0 表示手指朝上） */
  rotation: number
  /** 五指弯曲度 */
  curl: FiveCurl
  /** 手掌尺度 */
  scale: number
  /** 手部在世界坐标的深度（用于 z） */
  depth: number
}

/** 放松待机姿态（在两手之间过渡时使用） */
export const REST_HAND_POSE: HandPose = {
  wrist: [-0.42, 0.86],
  rotation: 0.35,
  curl: [0.25, 0.3, 0.32, 0.34, 0.36],
  scale: 1,
  depth: 0,
}

// 局部坐标（x 右 / y 下，手指朝 -y 延伸）下各指的掌指关节偏移与指节长度
const MCP_OFFSET: [number, number][] = [
  [-0.3, 0.06], // 拇指
  [-0.11, -0.27], // 食指
  [0.0, -0.3], // 中指
  [0.11, -0.28], // 无名指
  [0.21, -0.24], // 小指
]

const FINGER_DIR: [number, number][] = [
  [-0.76, -0.46], // 拇指
  [-0.16, -1.0], // 食指
  [0.0, -1.0], // 中指
  [0.17, -0.99], // 无名指
  [0.34, -0.94], // 小指
]

const FINGER_LENGTH: [number, number, number][] = [
  [0.17, 0.1, 0.08], // 拇指 MCP->CMC->IP->TIP 近似
  [0.2, 0.13, 0.09], // 食指
  [0.22, 0.14, 0.09], // 中指
  [0.2, 0.13, 0.09], // 无名指
  [0.16, 0.1, 0.08], // 小指
]

/** 每根手指一个 entail：習 outermost gain 系数（越大弯曲越明显） */
const CURL_GAIN = [0.85, 1.15, 1.15, 1.15, 1.0]

function rot2(x: number, y: number, a: number): [number, number] {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [c * x - s * y, s * x + c * y]
}

/** 由 HandPose 生成主手 21 点（局部->世界：先旋 transform，再平移） */
export function buildHand(pose: HandPose): Point3[] {
  const pts: Point3[] = []
  const [wx, wy] = pose.wrist
  pts.push([wx, wy, pose.depth]) // 0 手腕

  for (let f = 0; f < 5; f += 1) {
    let dx = MCP_OFFSET[f][0] * pose.scale
    let dy = MCP_OFFSET[f][1] * pose.scale
    ;[dx, dy] = rot2(dx, dy, pose.rotation)

    let px = wx + dx
    let py = wy + dy
    // 掌指关节（MCP）本身也是一个关键点：1-4 拇指 / 5-8 食指 ... 的起点
    pts.push([px, py, pose.depth])

    const base = rot2(FINGER_DIR[f][0], FINGER_DIR[f][1], pose.rotation)
    let dirx = base[0]
    let diry = base[1]
    const curl = Math.min(1, Math.max(0, pose.curl[f]))
    const gain = CURL_GAIN[f]

    for (let j = 0; j < 3; j += 1) {
      const len = FINGER_LENGTH[f][j] * pose.scale
      px += dirx * len
      py += diry * len
      // 每个关节按弯曲度向内旋转，形成自然的蜷曲
      ;[dirx, diry] = rot2(dirx, diry, curl * gain * 0.62)
      pts.push([px, py, pose.depth - j * 0.012])
    }
  }
  return pts
}

/** 生成上半身 6 点（静止：归一化空间的固定骨架参考系） */
export function buildPose(): Point3[] {
  return [
    [0, 0, 0], // 21 鼻（归一化原点）
    [-0.5, 0.3, 0], // 22 左肩
    [0.5, 0.3, 0], // 23 右肩
    [-0.76, 0.86, 0.02], // 24 左肘
    [0.76, 0.86, 0.02], // 25 右肘
    [-0.6, 1.3, 0.05], // 26 左腕
  ]
}

/** 由手部姿态合成完整一帧 27 点骨架 */
export function buildFrame(hand: HandPose): SkeletonFrame {
  return [...buildHand(hand), ...buildPose()]
}

// ---------------------------------------------------------------------------
// 姿态插值（骨架动画补间）
// ---------------------------------------------------------------------------

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** 缓入缓出，避免机械式线性运动 */
export function smoothstep(t: number): number {
  const c = Math.min(1, Math.max(0, t))
  return c * c * (3 - 2 * c)
}

/** 在两个手部姿态之间插值（含 shortest-path 角度插值） */
export function lerpHandPose(a: HandPose, b: HandPose, t: number): HandPose {
  let da = b.rotation - a.rotation
  while (da > Math.PI) da -= Math.PI * 2
  while (da < -Math.PI) da += Math.PI * 2
  return {
    wrist: [lerp(a.wrist[0], b.wrist[0], t), lerp(a.wrist[1], b.wrist[1], t)],
    rotation: a.rotation + da * t,
    curl: [
      lerp(a.curl[0], b.curl[0], t),
      lerp(a.curl[1], b.curl[1], t),
      lerp(a.curl[2], b.curl[2], t),
      lerp(a.curl[3], b.curl[3], t),
      lerp(a.curl[4], b.curl[4], t),
    ],
    scale: lerp(a.scale, b.scale, t),
    depth: lerp(a.depth, b.depth, t),
  }
}

/** 计算一整段序列的包围盒，供 SVG 自适应缩放（避免硬编码坐标范围） */
export function framesBounds(frames: SkeletonFrame[]) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const fr of frames) {
    for (const [x, y] of fr) {
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
    }
  }
  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: 1, maxY: 1 }
  }
  return { minX, minY, maxX, maxY }
}
