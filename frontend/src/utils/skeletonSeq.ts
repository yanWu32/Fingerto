// 把「逐 gloss 手部姿态」合成为真实的 (T, 27, 3) 骨架序列。
// 生成的序列形状/布局与 Fingerto/perception/skeleton.py 的契约一致，
// 使前端演示播放的就是识别层真正的输入张量，而非假造的坐标。
//
// 时间轴按 recognition.glossResult.timestamps 对齐：
//   [起手REST] -> gloss0 保持段 -> 过渡 -> gloss1 保持段 -> ... -> [归位REST]

import { buildFrame, lerpHandPose, smoothstep, type HandPose, type SkeletonFrame } from './skeleton'
import type { DemoScenario, GlossPose } from '@/api/demoData'

/** 由 gloss 时间戳推导「每个 gloss 的取样帧」，插值生成整段 (T, 27, 3) 序列 */
export function synthesizeFrames(
  scenario: DemoScenario,
  numFrames: number,
  fps: number
): SkeletonFrame[] {
  const { poses } = scenario
  const timestamps = scenario.recognition.glossResult?.timestamps ?? []
  if (poses.length === 0) return []

  // 每个 gloss 的中点时刻（秒）-> 帧索引
  const centersSec: number[] =
    timestamps.length === poses.length
      ? timestamps.map(([s, e]: [number, number]) => (s + e) / 2)
      : poses.map(
          (_: GlossPose, i: number) => (i + 0.5) * (scenario.sense.durationSec / poses.length)
        )

  const centerFrames = centersSec.map((t: number) => Math.round(t * fps))

  // 关键帧序列：起手 REST -> 各 gloss -> 归位 REST
  const keyframes: { frame: number; hand: HandPose }[] = [
    { frame: 0, hand: REST_HAND },
    ...poses.map((p: GlossPose, i: number) => ({
      frame: clampFrame(centerFrames[i] ?? (i + 1) * 10, numFrames),
      hand: p.hand,
    })),
    { frame: numFrames - 1, hand: REST_HAND },
  ]

  const frames: SkeletonFrame[] = []
  for (let t = 0; t < numFrames; t += 1) {
    // 找到 t 落在哪两个关键帧之间
    let i = 0
    while (i < keyframes.length - 2 && t >= keyframes[i + 1].frame) i += 1
    const a = keyframes[i]
    const b = keyframes[i + 1]
    const span = Math.max(1, b.frame - a.frame)
    const tNorm = smoothstep((t - a.frame) / span)
    frames.push(buildFrame(lerpHandPose(a.hand, b.hand, tNorm)))
  }
  return frames
}

function clampFrame(f: number, max: number): number {
  return Math.min(max - 1, Math.max(1, f))
}

/** 待机/起手姿态（五指自然微屈） */
const REST_HAND: HandPose = {
  wrist: [-0.42, 0.86],
  rotation: 0.35,
  curl: [0.25, 0.3, 0.32, 0.34, 0.36],
  scale: 1,
  depth: 0,
}

/** 序列形状描述文本，用于 UI 展示 (T, 27, 3) */
export function shapeText(frames: SkeletonFrame[]): string {
  return `(${frames.length}, 27, 3)`
}

export type { GlossPose }
