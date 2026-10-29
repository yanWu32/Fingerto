# 后端改动请求：智能体层接入真实 LLM（消除 mock 翻译）

> 触发背景：用户要求「页面结果必须来自后端真实输出，禁止前端编造」。
> 实测 `app/server.py` 的 `/pipeline/stream` 接口已能产出**真实识别层**（gloss 序列），
> 但**智能体层 `natural_language` / `intent` / `task` 是假的**（默认走 `_fake_llm`）。
> 这是后端行为，按约定前端侧不擅自改后端，故提出本改动请求，待确认后由后端负责人落地。

## 当前问题（已实测复现）
`POST /api/pipeline/stream` 返回：
```json
{ "type": "agent_done",
  "agent": { "natural_language": "（mock 翻译）用户用手语表达了一段内容。",
             "intent": "chat", "task": "none", "clarification": true,
             "low_confidence_words": ["40","30","万","五十"] } }
```
`natural_language` 是 `_fake_llm()` 写死的占位文案，并非真实翻译。

## 根因
`app/server.py`：
- `_use_fake_llm()` 默认返回 `True`（仅当环境变量 `DEEPSEEK_API_KEY` 存在时才返回 `False`）。
- `pipeline_stream` 中：`fake = _fake_llm if _use_fake_llm() else None`（line ~463），
  把 `fake` 传给 `_run_agent(..., fake)`，进而 `build_agent(agent_cfg, fake_completion=fake)`，
  最终 `LLMClient.complete()` 因 `self._fake is not None` 而返回假文案。

## 真实路径其实已就绪（无需重写）
`agent/llm_client.py` 的 `LLMClient` 已完整实现 OpenAI 兼容真实调用：
- `provider: deepseek`（默认），`base_url: https://api.deepseek.com/v1`，`model: deepseek-chat`；
- `complete()` 在 `self._fake is None` 时走 `client.chat.completions.create(...)`；
- API Key 从环境变量 `DEEPSEEK_API_KEY` 读取（`configs/agent.yaml` 的 `api_key_env.deepseek`）。
只需让 `app/server.py` 在「有 Key」时不再注入 `fake`，即可走真实翻译。

## 建议改动（最小、向后兼容）
在 `app/server.py` 中，将「是否使用 fake」的判定收紧，并补充明确开关：

1. 新增环境变量开关（优先级最高）：
   - `FINGERTO_AGENT_FAKE=1` → 强制 fake（保留用于无 Key 自测 / CI）；
   - 未设置且 `DEEPSEEK_API_KEY` 存在 → 走真实 LLM；
   - 未设置且 `DEEPSEEK_API_KEY` 缺失 → **明确报错**，而不是悄悄用假数据糊弄前端。
2. 修改 `_use_fake_llm()` 逻辑：
   ```python
   def _use_fake_llm() -> bool:
       if os.environ.get("FINGERTO_AGENT_FAKE") == "1":
           return True
       # 有 Key 走真实；无 Key 则明确失败（不再默认 fake）
       return False   # 或：raise 在调用处
   ```
   并在 `pipeline_stream` 调用 `_run_agent` 前校验：若需要用真实 LLM 但 `openai` 未装或 Key 缺失，
   返回 503 + `msg="智能体需要 DEEPSEEK_API_KEY，请配置后重试"`（前端已能透出该 msg）。
3. 不改变 `/pipeline/stream` 的事件结构与字段（前端已按当前 schema 实现，无需前端改动）。

## 不改动的部分
- 识别层（gloss 序列）保持真实，不动。
- 前端无需任何修改（页面已只展示后端真实事件）。
- `configs/agent.yaml` 的 `provider/model/endpoint` 已正确，无需改。

## 验收
1. 配置 `DEEPSEEK_API_KEY=sk-...` 后 `python serve.py --backend-fastapi`；
2. 上传 `video_01.mp4` 跑 `/api/pipeline/stream`；
3. `agent_done` 的 `natural_language` 应为**真实中文翻译**而非「（mock 翻译）…」，
   `intent`/`task` 应反映视频语义；`low_confidence_words` 仍由真实置信度得出。
4. 未配置 Key 时：`agent_done` 不应出现，接口应返回 503 且 `msg` 说明缺 Key。

## 备注
前端侧已做的配套（已提交）：
- `RecognizeView` 已移除静默 mock 降级，后端不可用/缺依赖时直接报错；
- `AgentSteps` 在后端未下发分步数据时显式标注「后端未提供分步数据」，不编造三步提示词；
- `serve.py` 默认即启动 FastAPI 真实后端（已修复此前误启 Gradio 的 bug）。
