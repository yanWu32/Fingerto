// v-html 高亮渲染的安全工具。
// DemoView 需要把 Step1/2/3 的 prompt 与输出中的关键 token（如 <lowconf:...>、
// 任务标识）包上 <mark> 高亮。输出到 v-html 的字符串必须先转义 HTML 特殊字符，
// 再插入我们自己的标签，杜绝注入内容成为可执行 HTML（XSS 红线，规范 7.1）。
// 本模块有对应单元测试：sanitize.test.ts。

/** 转义 HTML 特殊字符 */
export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/**
 * 把 text 转义后，将其中的 tokens 逐个包上 <mark class="tk">。
 * 注意：token 自身也会被转义后再匹配，因此恶意 token 不会引入可执行标签。
 */
export function highlightTokens(text: string, tokens: string[]): string {
  let escaped = escapeHtml(text)
  for (const tk of tokens) {
    if (!tk) continue
    const escTk = escapeHtml(tk)
    // 先收集所有匹配位置，再从后往前插入，避免前面的插入影响后面的下标
    const idxs: number[] = []
    let from = 0
    for (;;) {
      const i = escaped.indexOf(escTk, from)
      if (i === -1) break
      idxs.push(i)
      from = i + escTk.length
    }
    for (let k = idxs.length - 1; k >= 0; k -= 1) {
      const i = idxs[k]
      escaped =
        escaped.slice(0, i) +
        `<mark class="tk">${escaped.slice(i, i + escTk.length)}</mark>` +
        escaped.slice(i + escTk.length)
    }
  }
  return escaped
}
