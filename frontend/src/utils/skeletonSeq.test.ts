import { describe, it, expect } from 'vitest'
import { synthesizeFrames, shapeText } from './skeletonSeq'
import { buildScenario } from '../api/demoPipeline'
import {
  NUM_POINTS,
  buildFrame,
  buildHand,
  buildPose,
  framesBounds,
  lerpHandPose,
  REST_HAND_POSE,
  type HandPose,
} from './skeleton'

// 骨架生成器的正确性：前端演示播放的就是识别层真正的输入张量，
// 因此这里必须按 tests/contract.py 的骨架契约校验形状与数值。
describe('27 点骨架模型', () => {
  it('buildFrame 产出恰好 27 个点', () => {
    const f = buildFrame(REST_HAND_POSE)
    expect(f).toHaveLength(NUM_POINTS)
    expect(f).toHaveLength(27)
  })

  it('buildHand 产出主手 21 点，每点为 3 维', () => {
    const hand = buildHand(REST_HAND_POSE)
    expect(hand).toHaveLength(21)
    for (const p of hand) {
      expect(p).toHaveLength(3)
      expect(Number.isFinite(p[0])).toBe(true)
      expect(Number.isFinite(p[1])).toBe(true)
      expect(Number.isFinite(p[2])).toBe(true)
    }
  })

  it('buildPose 产出上半身 6 点（鼻为归一化原点）', () => {
    const pose = buildPose()
    expect(pose).toHaveLength(6)
    expect(pose[0]).toEqual([0, 0, 0]) // 鼻
  })

  it('lerpHandPose 在 t=0 / t=1 时分别还原两端姿态', () => {
    const a: HandPose = { wrist: [0, 0], rotation: 0, curl: [0, 0, 0, 0, 0], scale: 1, depth: 0 }
    const b: HandPose = { wrist: [1, 1], rotation: 1, curl: [1, 1, 1, 1, 1], scale: 2, depth: 0.5 }
    expect(lerpHandPose(a, b, 0).wrist).toEqual([0, 0])
    expect(lerpHandPose(a, b, 1).wrist).toEqual([1, 1])
    expect(lerpHandPose(a, b, 1).scale).toBe(2)
  })
})

describe('synthesizeFrames：合成 (T, 27, 3)', () => {
  it('产出帧数等于视频帧数，且每帧 27 点 x 3 维', () => {
    const scenario = buildScenario(0)
    const frames = synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
    expect(frames).toHaveLength(scenario.sense.numFrames)
    expect(frames).toHaveLength(72)
    for (const fr of frames) {
      expect(fr).toHaveLength(27)
      for (const p of fr) expect(p).toHaveLength(3)
    }
  })

  it('形状文本符合契约 (T, 27, 3)', () => {
    const scenario = buildScenario(0)
    const frames = synthesizeFrames(scenario, 72, 30)
    expect(shapeText(frames)).toBe('(72, 27, 3)')
  })

  it('无 NaN / Infinity（契约要求 finite）', () => {
    const scenario = buildScenario(0)
    const frames = synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
    for (const fr of frames) {
      for (const p of fr) {
        for (const v of p) {
          expect(Number.isFinite(v)).toBe(true)
        }
      }
    }
  })

  it('首帧与末帧均回到待机姿态附近（起手/归位）', () => {
    const scenario = buildScenario(0)
    const frames = synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
    const first = frames[0]
    const last = frames[frames.length - 1]
    // 待机手腕约在 (-0.42, 0.86)
    expect(Math.abs(first[0][0] + 0.42)).toBeLessThan(0.05)
    expect(Math.abs(last[0][1] - 0.86)).toBeLessThan(0.05)
  })

  it('包围盒有界，可供 SVG 自适应 viewBox', () => {
    const scenario = buildScenario(0)
    const frames = synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
    const b = framesBounds(frames)
    expect(Number.isFinite(b.minX)).toBe(true)
    expect(b.maxX).toBeGreaterThan(b.minX)
    expect(b.maxY).toBeGreaterThan(b.minY)
  })
})

describe('演示数据与真实代码的一致性', () => {
  it('低置信词严格由 clarify_threshold=0.4 决定', () => {
    const scenario = buildScenario(0)
    const g = scenario.recognition.glossResult!
    const low = g.gloss_sequence.filter((_, i) => g.confidence[i] < 0.4)
    expect(scenario.steps.agentResult!.low_confidence_words).toEqual(low)
    expect(low).toEqual(['冷'])
  })

  it('Step1 对 conf<0.75 的词标注 <lowconf:>', () => {
    const scenario = buildScenario(0)
    const prompt = scenario.steps.steps[0].promptUser
    expect(prompt).toContain('<lowconf:冷:0.34>')
    // 0.89 虽高但低于 high_threshold=0.75? 否 -> 不应标注
    expect(prompt).not.toContain('<lowconf:窗')
  })

  it('Step3 的 task 只能取自真实动作表', () => {
    const scenario = buildScenario(0)
    const allowed = [
      'turn_on_light',
      'turn_off_light',
      'turn_on_ac',
      'turn_off_ac',
      'open_window',
      'close_window',
      'open_curtain',
      'close_curtain',
      'none',
    ]
    expect(allowed).toContain(scenario.steps.agentResult!.task)
  })

  it('gloss / confidence / timestamps 三者等长（GLOSS_SCHEMA 要求）', () => {
    const scenario = buildScenario(0)
    const g = scenario.recognition.glossResult!
    expect(g.confidence).toHaveLength(g.gloss_sequence.length)
    expect(g.timestamps).toHaveLength(g.gloss_sequence.length)
  })
})
