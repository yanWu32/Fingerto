import { describe, it, expect } from 'vitest'
import { runPipeline } from './pipeline'
import { mockPipeline } from './mock'
// 使用相对路径而非 '@/' 别名：使测试不依赖 vite/vitest 配置的 alias 解析，
// 任何 runner 下都能直接跑（也便于在没有构建工具的环境里验证）。
import { USE_MOCK } from '../utils/env'

// pipeline 串联测试：mock 模式下保证识别层 + 智能体层契约结构正确，
// 为后续切换真实后端提供回归基线。
describe('pipeline (mock)', () => {
  it('返回识别 + 智能体 两段结果', async () => {
    const res = await runPipeline()
    expect(Array.isArray(res.recognition.gloss_sequence)).toBe(true)
    expect(res.recognition.gloss_sequence.length).toBe(res.recognition.confidence.length)
    expect(res.recognition.gloss_sequence.length).toBe(res.recognition.timestamps.length)
    expect(typeof res.agent.natural_language).toBe('string')
    expect(typeof res.agent.task).toBe('string') // 红线：task 必为 string
    expect(typeof res.agent.clarification).toBe('boolean') // 红线：clarification 必为 boolean
  })

  it('mock 数据与契约一致', async () => {
    const m = await mockPipeline()
    expect(m.agent.task).toBe('close_window')
    expect(m.agent.clarification).toBe(false)
  })
})

// 明确声明当前测试在 mock 模式下运行，接真实后端后该用例应重新评估
describe('env', () => {
  it('默认处于 mock 模式（接后端前）', () => {
    expect(USE_MOCK).toBe(true)
  })
})
