// 工具函数：数据格式化展示（规范 1.1 前端职责：日期/文本格式化）

export function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(0)}%`
}

export function formatTimeRange(range: [number, number]): string {
  return `${range[0].toFixed(1)}s – ${range[1].toFixed(1)}s`
}
