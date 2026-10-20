import { describe, it, expect } from 'vitest'
import { highlightTokens, escapeHtml } from './sanitize'

// highlightTokens 的结果会交给 v-html，因此转义是安全红线：必须先转义用户输入，
// 再插入我们自己的 <mark> 标签，杜绝注入的 HTML 被执行（规范 7.1）。

describe('escapeHtml', () => {
  it('转义三种 HTML 特殊字符', () => {
    expect(escapeHtml('<')).toBe('&lt;')
    expect(escapeHtml('>')).toBe('&gt;')
    expect(escapeHtml('&')).toBe('&amp;')
  })

  it('引号不转义（属性值由模板插值而非属性上下文使用）', () => {
    expect(escapeHtml('"hi"')).toBe('"hi"')
  })
})

describe('highlightTokens XSS 安全性', () => {
  it('转义 HTML 特殊字符，注入的 <script> 不会成为可执行标签', () => {
    const out = highlightTokens('<script>alert(1)</script>', [])
    expect(out).not.toContain('<script>')
    expect(out).toContain('&lt;script&gt;')
  })

  it('转义 img onerror 等事件属性载荷', () => {
    const out = highlightTokens('<img src=x onerror=alert(1)>', [])
    expect(out).not.toContain('<img')
    expect(out).toContain('&lt;img')
  })

  it('转义 & 符号，避免实体吞并', () => {
    expect(highlightTokens('a & b', [])).toContain('a &amp; b')
  })

  it('高亮目标 token 本身能正确包上 <mark>', () => {
    const out = highlightTokens('当前 gloss：我 <lowconf:冷:0.34>', ['<lowconf:'])
    expect(out).toContain('<mark class="tk">&lt;lowconf:</mark>')
  })

  it('恶意 token 也被转义后再包 <mark>，不引入可执行标签', () => {
    const out = highlightTokens('payload <script>x</script>', ['<script>'])
    expect(out).not.toContain('<script>')
    expect(out).toContain('<mark class="tk">&lt;script&gt;</mark>')
  })
})
