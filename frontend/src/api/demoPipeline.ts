// 端到端「手语视频 -> 文本」演示编排。
// 按真实分层依次产出各阶段快照：感知 -> 骨架 -> 切分 -> 识别 -> 智能体三步 -> 应用执行。
// 所有 stage 由 VITE_USE_MOCK 决定是否走本地 mock 数据；真实后端就绪后可替换为 /api 请求。
//
// 与真实代码对齐的关键点（agent/pipeline.py + configs/agent.yaml）：
//   clarify_threshold = 0.4   低于此值才进 low_confidence_words
//   high_threshold    = 0.75  低于此值在 Step1 被标注 <lowconf:词:分>
//   Step3 的 task 取自 intent.actions，无任务时为 'none'

import { USE_MOCK } from '@/utils/env'
import { DEMO_SCENARIOS, type DemoScenario } from './demoData'
import { synthesizeFrames } from '@/utils/skeletonSeq'
import type { AgentResult, GlossResult } from './types'

/** 流水线阶段 */
export type PipelineStageName =
  | 'idle'
  | 'sensing'
  | 'skeleton'
  | 'segmenting'
  | 'recognizing'
  | 'agentStep1'
  | 'agentStep2'
  | 'agentStep3'
  | 'executing'
  | 'done'

/** 一帧 interstitial 快照 */
export interface StageSnapshot {
  stage: PipelineStageName
  label: string
  layer: string
  detail: string
  payload?: unknown
}

/** 演示最终产物 */
export interface DemoOutput {
  gloss: GlossResult
  agent: AgentResult
  scenario: DemoScenario
}

export interface AgentStepOutput {
  step: 1 | 2 | 3
  title: string
  promptSystem: string
  promptUser: string
  output: string
  highlightTokens: string[]
}

/** 取默认演示场景（并为其生成骨架序列） */
export function buildScenario(index = 0): DemoScenario {
  const scenario = DEMO_SCENARIOS[index] ?? DEMO_SCENARIOS[0]
  if (scenario.frames.length === 0) {
    scenario.frames = synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
  }
  return scenario
}

/** 可用场景列表 */
export function listScenarios(): { id: string; title: string; summary: string }[] {
  return DEMO_SCENARIOS.map((s) => ({ id: s.id, title: s.title, summary: s.summary }))
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * 逐步执行整条流水线，每个阶段回调一次（支持「单步播放」与自动播放）。
 * 所有阶段耗时在真实场景由后端决定；mock 下用 delay 模拟处理耗时。
 */
export async function runFullDemo(
  scenario: DemoScenario,
  onStage: (s: StageSnapshot) => void,
  stepDelay = 520
): Promise<DemoOutput> {
  if (!USE_MOCK) {
    // 真实后端尚未提供 /api/* 的流水线串流；此处保留降级：直接返回结果而不阻塞 UI
    onStage({
      stage: 'done',
      label: '真实后端模式',
      layer: '应用层',
      detail: 'VITE_USE_MOCK=false 时由后端 /api/pipeline 返回结果',
    })
  }

  onStage({
    stage: 'sensing',
    label: '读取视频帧',
    layer: '感知层',
    detail: `${scenario.sense.source} · ${scenario.sense.width}×${scenario.sense.height} · ${scenario.sense.fps}fps`,
  })
  await delay(stepDelay)

  const frames = scenario.frames.length
    ? scenario.frames
    : synthesizeFrames(scenario, scenario.sense.numFrames, scenario.sense.fps)
  scenario.frames = frames

  onStage({
    stage: 'skeleton',
    label: '提取 27 点骨架',
    layer: '感知层',
    detail: `${scenario.sense.extractor} → 张量 ${scenario.skeleton.shape}`,
    payload: {
      numFrames: frames.length,
      first: frames[0]?.slice(0, 3),
      last: frames[frames.length - 1]?.slice(0, 3),
    },
  })
  await delay(stepDelay)

  onStage({
    stage: 'segmenting',
    label: '运动能量切分 + 滑动窗口',
    layer: '切分层',
    detail: `${scenario.segmentation.numSegments} 个活跃段 → ${scenario.segmentation.numWindows} 个识别窗口`,
    payload: scenario.segmentation,
  })
  await delay(stepDelay)

  const gloss = scenario.recognition.glossResult!
  onStage({
    stage: 'recognizing',
    label: '逐窗口推理并投票聚合',
    layer: '识别层',
    detail: `${scenario.recognition.model} · gloss 序列 ${gloss.gloss_sequence.length} 词`,
    payload: {
      windows: scenario.recognition.windows,
      glossResult: gloss,
    },
  })
  await delay(stepDelay)

  // ---- 智能体三步 ----
  const s1 = scenario.steps.steps[0]
  onStage({
    stage: 'agentStep1',
    label: s1.title,
    layer: '智能体层',
    detail: '低置信词因低置信触发消歧',
    payload: { step: s1, highlight: scenario.steps.highlightTokens },
  })
  await delay(stepDelay)

  const s2 = scenario.steps.steps[1]
  onStage({
    stage: 'agentStep2',
    label: s2.title,
    layer: '智能体层',
    detail: 'gloss 序列 → 通顺中文',
    payload: { step: s2, highlight: scenario.steps.highlightTokens },
  })
  await delay(stepDelay)

  const s3 = scenario.steps.steps[2]
  onStage({
    stage: 'agentStep3',
    label: s3.title,
    layer: '智能体层',
    detail: '意图识别 + 任务生成',
    payload: { step: s3, highlight: scenario.steps.highlightTokens },
  })
  await delay(stepDelay)

  const agent = scenario.steps.agentResult!
  onStage({
    stage: 'executing',
    label: scenario.execution.noop ? '无需执行（无任务）' : scenario.execution.taskName,
    layer: '应用层',
    detail: scenario.execution.noop
      ? 'task 为 none，不触发设备控制'
      : `${scenario.execution.apiPath} · ${scenario.execution.controllerStatus}`,
    payload: {
      execution: scenario.execution,
      envBefore: scenario.envBefore,
      envAfter: scenario.envAfter,
    },
  })
  await delay(stepDelay)

  onStage({
    stage: 'done',
    label: '完成：手语视频 → 文本 → 任务',
    layer: '全链路',
    detail: `最终文本「${agent.natural_language}」`,
    payload: { gloss, agent },
  })

  return { gloss, agent, scenario }
}

/** 取出某场景的智能体三步详情（供 UI 展示 prompt / 输出） */
export function getAgentSteps(scenario: DemoScenario): AgentStepOutput[] {
  return scenario.steps.steps.map((s, i) => ({
    step: (i + 1) as 1 | 2 | 3,
    title: s.title,
    promptSystem: s.promptSystem,
    promptUser: s.promptUser,
    output: s.output,
    highlightTokens: scenario.steps.highlightTokens,
  }))
}
