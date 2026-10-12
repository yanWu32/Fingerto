// 前端 mock 数据：后端未就绪时用于跑通界面（规范 1.3 / 方案 4.2 数据无关可先行）。
// 接口契约与 Fingerto/tests/contract.py 一致，后续直接替换为真实 /api 请求。
// 类型统一引用 types.ts，避免与后端契约漂移。
import type { AgentResult, GlossResult, PipelineResult } from './types'

export function mockRecognition(): Promise<GlossResult> {
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

export function mockPipeline(): Promise<PipelineResult> {
  return Promise.all([mockRecognition(), mockAgent()]).then(([recognition, agent]) => ({
    recognition,
    agent,
  }))
}
