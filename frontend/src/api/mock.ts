// 前端 mock 数据：后端未就绪时用于跑通界面（规范 1.3 / 方案 4.2 数据无关可先行）
// 接口契约与方案第 4 节 JSON 一致，后续直接替换为真实 /api 请求。

// 识别层输出（识别 → 智能体 的契约）
export interface RecognitionResult {
  gloss_sequence: string[]
  confidence: number[]
  timestamps: [number, number][]
}

// 智能体层输出（契约见 Fingerto/tests/contract.py AGENT_SCHEMA）
// 注意：task 必须是 string（无任务时为 'none'），clarification 必须是 bool，二者不可为 null。
export interface AgentResult {
  natural_language: string
  intent: string
  task: string
  clarification: boolean
  low_confidence_words: string[]
}

export function mockRecognition(): Promise<RecognitionResult> {
  return Promise.resolve({
    gloss_sequence: ['我', '冷', '窗', '关'],
    confidence: [0.92, 0.58, 0.91, 0.85],
    timestamps: [
      [0.0, 0.4],
      [0.5, 0.9],
      [1.0, 1.4],
      [1.5, 1.9],
    ],
  })
}

export function mockAgent(): Promise<AgentResult> {
  return Promise.resolve({
    natural_language: '我觉得冷，能关一下窗户吗？',
    intent: 'command',
    task: 'close_window',
    clarification: false,
    low_confidence_words: ['冷'],
  })
}
