// 端到端演示数据：视频 -> 骨架 -> 切分 -> 识别 -> 智能体三步 -> 应用执行。
// 契约事实源：Fingerto/tests/contract.py（GLOSS_SCHEMA / AGENT_SCHEMA）
// 业务事实源：Fingerto/agent/pipeline.py（三步提示管线）、Fingerto/configs/agent.yaml（阈值与动作表）
// mock 数据与真实代码行为必须保持一致（阈值不符即为 bug）：
//   - 低置信路由：置信度 < clarify_threshold(0.4) 的词才进入 low_confidence_words
//   - Step1 消歧：对 conf < 0.75 的词标注 <lowconf:词:分数> 后再送入 LLM
//   - Step3 决策：task 只能取自 intent.actions（无任务时为 'none'）
// 说明：本文件内所有字符串均为「原始文本」，由模板插值负责转义，故不写 HTML 实体。

import type { AgentResult, GlossResult } from './types'
import type { SkeletonFrame, HandPose } from '@/utils/skeleton'

/** 单个滑动窗口：模型真正推理的输入单元 */
export interface SkeletonWindow {
  startFrame: number
  endFrame: number
  startTime: number
  endTime: number
}

/** 感知层：视频 -> 骨架 张量 */
export interface SenseStage {
  source: string
  width: number
  height: number
  fps: number
  durationSec: number
  numFrames: number
  extractor: string
}

/** 骨架张量本身 */
export interface SkeletonStage {
  numPoints: number
  layout: string
  numFrames: number
  shape: string
  channels: string
}

/** 切分层：能量粗切分 + 滑动窗口 */
export interface SegmentStage {
  method: string
  energyReference: string
  velocityThreshold: number
  minStaticFrames: number
  windowSize: number
  stride: number
  smoothing: string
  numSegments: number
  numWindows: number
  segments: { startTime: number; endTime: number }[]
}

/** 单个窗口的推理产出（多窗口投票前的原始结果） */
export interface WindowResult {
  startTime: number
  endTime: number
  gloss: string
  confidence: number
}

/** 识别层 */
export interface RecognitionStage {
  model: string
  inputShape: string
  windows: WindowResult[]
  numWindows: number
  glossResult: GlossResult | null
}

/** 智能体三步中的一步 */
export interface TranslationStep {
  name: string
  title: string
  promptSystem: string
  promptUser: string
  output: string
}

/** 智能体三步管线 */
export interface PipelineSteps {
  confirmEnabled: boolean
  translateEnabled: boolean
  decideEnabled: boolean
  steps: TranslationStep[]
  /** 展示时需要高亮的 token（如 <lowconf:...>、任务标识） */
  highlightTokens: string[]
  agentResult: AgentResult | null
}

/** 模拟智能家居环境状态 */
export interface EnvironmentSnapshot {
  windowOpen: number
  curtainOpen: number
  airConditionerTemperature: number
  lightStatus: string
  lastIntent: string
}

/** 应用层任务执行 */
export interface TaskExecution {
  apiPath: string
  taskName: string
  taskId: string
  controllerStatus: string
  changes: { label: string; from: string; to: string }[]
  noop: boolean
}

/** 手部姿态关键帧：用于合成 (T, 27, 3) 演示序列 */
export interface GlossPose {
  gloss: string
  hand: HandPose
}

/** 一个完整演示场景 */
export interface DemoScenario {
  id: string
  title: string
  summary: string
  /** 该场景是否为「需要澄清」路线 */
  clarifies: boolean
  sense: SenseStage
  skeleton: SkeletonStage
  segmentation: SegmentStage
  recognition: RecognitionStage
  steps: PipelineSteps
  envBefore: EnvironmentSnapshot
  envAfter?: EnvironmentSnapshot
  execution: TaskExecution
  /** 逐 gloss 的手部关键姿态，用于生成骨架序列 */
  poses: GlossPose[]
  /** 合成后的整段骨架序列；(T, 27, 3)，会就地填充 */
  frames: SkeletonFrame[]
}

/** 休息/过渡姿态（起手与归位） */
const REST_POSE: HandPose = {
  wrist: [-0.42, 0.86],
  rotation: 0.35,
  curl: [0.25, 0.3, 0.32, 0.34, 0.36],
  scale: 1,
  depth: 0,
}

/** 五指张开（手背朝向镜头、指尖朝上），用于「窗」这类开放手型 */
function openHand(): [number, number, number, number, number] {
  return [0.1, 0.05, 0.05, 0.06, 0.08]
}

const COLD_WINDOW_SCENARIO: DemoScenario = {
  id: 'cold-window',
  title: '① “我感觉有点冷，能关一下窗户吗” → 智能家居控制',
  summary:
    '命中 command 意图，Step3 决策出 close_window 任务；其中「冷」的置信度 0.34 低于澄清阈值 0.4，触发 Step1 消歧。',
  clarifies: false,

  sense: {
    source: 'data/demo/cold_window.mp4',
    width: 1280,
    height: 720,
    fps: 30,
    durationSec: 2.4,
    numFrames: 72,
    extractor: 'MediaPipe Hands(21) + Pose(6)',
  },

  skeleton: {
    numPoints: 27,
    layout: '主手 21 点 + 上半身 6 点',
    numFrames: 72,
    shape: '(72, 27, 3)',
    channels: 'x / y / z 归一化坐标',
  },

  segmentation: {
    method: '运动能量粗切分 + 滑动窗口细识别',
    energyReference: '左腕（第 26 点）帧间速度',
    velocityThreshold: 0.02,
    minStaticFrames: 5,
    windowSize: 30,
    stride: 10,
    smoothing: 'vote',
    numSegments: 3,
    numWindows: 8,
    segments: [
      { startTime: 0.0, endTime: 0.4 },
      { startTime: 0.43, endTime: 1.2 },
      { startTime: 1.27, endTime: 2.37 },
    ],
  },

  recognition: {
    model: 'ST-GCN + Transformer 时序池化',
    inputShape: '(30, 27, 3)',
    // 8 个滑动窗口的原始产出；重叠窗口的同 gloss 经 vote 投票合并后得到下方 glossResult
    windows: [
      { startTime: 0.0, endTime: 0.4, gloss: '我', confidence: 0.72 },
      { startTime: 0.43, endTime: 0.73, gloss: '感觉', confidence: 0.89 },
      { startTime: 0.53, endTime: 0.83, gloss: '感觉', confidence: 0.86 },
      { startTime: 1.27, endTime: 1.57, gloss: '有点', confidence: 0.81 },
      { startTime: 1.4, endTime: 1.7, gloss: '冷', confidence: 0.34 },
      { startTime: 1.63, endTime: 1.93, gloss: '窗', confidence: 0.91 },
      { startTime: 1.9, endTime: 2.2, gloss: '关', confidence: 0.88 },
      { startTime: 2.07, endTime: 2.37, gloss: '关', confidence: 0.79 },
    ],
    numWindows: 8,
    glossResult: {
      gloss_sequence: ['我', '感觉', '有点', '冷', '窗', '关'],
      confidence: [0.72, 0.89, 0.81, 0.34, 0.91, 0.88],
      timestamps: [
        [0.0, 0.4],
        [0.43, 0.73],
        [1.27, 1.57],
        [1.4, 1.7],
        [1.63, 1.93],
        [1.9, 2.2],
      ],
    },
  },

  steps: {
    confirmEnabled: true,
    translateEnabled: true,
    decideEnabled: true,
    steps: [
      {
        name: 'step1_confirm',
        title: 'Step1 · 低置信消歧（Disambiguation）',
        promptSystem:
          '手语序列中有低置信词（已标注 <lowconf:词:分数>）。请结合上下文判断其最可能的词，若不确定请标记为需要澄清。只输出修正后的 gloss 序列，空格分隔，低置信词用尖括号标注你仍不确定的：<lowconf:词>。',
        promptUser:
          '当前 gloss：我 <lowconf:感觉:0.89> <lowconf:有点:0.81> <lowconf:冷:0.34> 窗 关',
        output: '我 感觉 有点 冷 窗 关',
      },
      {
        name: 'step2_translate',
        title: 'Step2 · Gloss → 自然语言（Translation）',
        promptSystem:
          '你是手语对话系统的智能体。用户输入是手语识别得到的 gloss 序列（手语词，可能含 <unk> 表示无法识别）。请把它翻译成自然、通顺的中文句子，保持原意，不要臆造未出现的信息。',
        promptUser: '对话上下文：\n\n手语 gloss：我 感觉 有点 冷 窗 关',
        output: '我感觉有点冷，能帮我关一下窗户吗？',
      },
      {
        name: 'step3_decide',
        title: 'Step3 · 意图识别 / 任务生成（Decision）',
        promptSystem:
          '根据对话上下文与用户提供/环境，判断用户意图并生成可执行任务。意图类型限定为：chat / query / command / clarify；若为 command，任务标识取自允许动作列表；否则 task 填 none。输出严格 JSON。',
        promptUser:
          '对话上下文：\n\n环境状态：{"window_open": 1, "curtain_open": 1, "ac_temperature": 26, "light": "on"}\n\n用户本轮表达：我感觉有点冷，能帮我关一下窗户吗？\n\n允许控制动作：turn_on_light, turn_off_light, turn_on_ac, turn_off_ac, open_window, close_window, open_curtain, close_curtain',
        output: '{"intent": "command", "task": "close_window", "need_clarify": false}',
      },
    ],
    highlightTokens: ['<lowconf:', 'close_window', '"intent": "command"'],
    agentResult: {
      natural_language: '我感觉有点冷，能帮我关一下窗户吗？',
      intent: 'command',
      task: 'close_window',
      clarification: false,
      low_confidence_words: ['冷'],
    },
  },

  envBefore: {
    windowOpen: 1,
    curtainOpen: 1,
    airConditionerTemperature: 26,
    lightStatus: 'on',
    lastIntent: '—',
  },
  envAfter: {
    windowOpen: 0,
    curtainOpen: 1,
    airConditionerTemperature: 26,
    lightStatus: 'on',
    lastIntent: 'command',
  },

  execution: {
    apiPath: 'POST /api/agent/task?intent=command&task=close_window',
    taskName: '关闭客厅窗户',
    taskId: 'close_window',
    controllerStatus: '在线（online）',
    changes: [
      { label: '窗户状态 window_open', from: '1（开启）', to: '0（关闭）' },
      { label: '窗帘状态 curtain_open', from: '1（拉开）', to: '1（保持不变）' },
      { label: '空调温度 ac_temperature', from: '26℃', to: '26℃（按偏好保留）' },
    ],
    noop: false,
  },

  poses: [
    {
      gloss: '我',
      hand: {
        wrist: [-0.3, 0.62],
        rotation: 0.1,
        curl: [0.2, 0.95, 0.95, 0.95, 0.95],
        scale: 1,
        depth: 0,
      },
    },
    {
      gloss: '感觉',
      hand: {
        wrist: [-0.26, 0.5],
        rotation: -0.15,
        curl: [0.3, 0.2, 0.15, 0.2, 0.25],
        scale: 1,
        depth: -0.02,
      },
    },
    {
      gloss: '有点',
      hand: {
        wrist: [-0.32, 0.58],
        rotation: 0.45,
        curl: [0.5, 0.4, 0.35, 0.4, 0.45],
        scale: 1,
        depth: 0.01,
      },
    },
    {
      gloss: '冷',
      hand: {
        wrist: [-0.28, 0.72],
        rotation: 0.2,
        curl: [0.6, 0.75, 0.8, 0.8, 0.82],
        scale: 0.98,
        depth: 0.03,
      },
    },
    {
      gloss: '窗',
      hand: { wrist: [-0.2, 0.46], rotation: 0.05, curl: openHand(), scale: 1.02, depth: -0.04 },
    },
    {
      gloss: '关',
      hand: {
        wrist: [-0.36, 0.68],
        rotation: 0.6,
        curl: [0.85, 0.9, 0.92, 0.92, 0.9],
        scale: 1,
        depth: 0.05,
      },
    },
  ],
  frames: [],
}

export const DEMO_SCENARIOS: DemoScenario[] = [COLD_WINDOW_SCENARIO]

export { REST_POSE }
