# ============================================================
# Fingerto 环境一键搭建脚本 (Windows / PowerShell)
#
# 目标：新建 conda 环境 fingerto（Python 3.10），装上项目所需依赖
#   1. conda create -n fingerto python=3.10
#   2. torch + torchvision (CUDA 12.1)
#   3. opencv-python / mediapipe / gradio / openai 等
#   4. 运行 tools/check_env.py 自检
#
# 用法（在 PowerShell 里）:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\tools\setup_env.ps1
#
# 前置条件：网络可用（需能访问 conda 源、PyPI 镜像、download.pytorch.org）
# ============================================================

$ErrorActionPreference = "Stop"
$ENV_NAME = "fingerto"
$REPO     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PY       = "E:\Program\anaconda\anaconda3\envs\$ENV_NAME\python.exe"

function Info($msg)  { Write-Host "`n[INFO] $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg)  { Write-Host "[FAIL] $msg" -ForegroundColor Red }

# ---------- 0. 前置检查 ----------
Info "0/4 前置检查"

# 网络检查
$netOk = $false
foreach ($u in @("https://pypi.org/simple/", "https://pypi.tuna.tsinghua.edu.cn/simple/")) {
    $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 15 $u 2>$null
    if ($code -eq "200") { $netOk = $true; Ok "网络可达: $u"; break }
}
if (-not $netOk) {
    Fail "网络不可达（PyPI 与清华源均连不上）。"
    Warn "请确认代理软件已启动（当前系统代理配置为 127.0.0.1:7897，但端口未监听）。"
    Warn "代理恢复后重新运行本脚本即可，脚本支持重复执行。"
    exit 1
}

# conda 检查
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Fail "未找到 conda，请确认 Anaconda 已安装并加入 PATH。"
    exit 1
}
Ok "conda 已就绪"

# ---------- 1. 创建环境 ----------
Info "1/4 创建 conda 环境 '$ENV_NAME' (Python 3.10)"
$exists = & conda env list 2>$null | Select-String -Pattern "^$ENV_NAME\s"
if ($exists) {
    Ok "环境 '$ENV_NAME' 已存在，跳过创建"
} else {
    & conda create -n $ENV_NAME python=3.10 -y
    if ($LASTEXITCODE -ne 0) { Fail "创建环境失败"; exit 1 }
    Ok "环境创建完成"
}

if (-not (Test-Path $PY)) {
    Fail "未找到解释器: $PY"
    exit 1
}
Ok "解释器: $PY"
& $PY --version

# ---------- 2. 安装 torch (CUDA 12.1) ----------
Info "2/4 安装 torch / torchvision (cu121) —— 约 2.5GB，耗时较长"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Warn "cu121 安装失败，回退到 PyPI 默认版本"
    & $PY -m pip install torch torchvision
    if ($LASTEXITCODE -ne 0) { Fail "torch 安装失败"; exit 1 }
}
Ok "torch 安装完成"

# ---------- 3. 安装其余依赖 ----------
Info "3/4 安装其余依赖 (opencv / mediapipe / gradio / openai ...)"
& $PY -m pip install -r "$REPO\requirements.txt"
if ($LASTEXITCODE -ne 0) { Warn "部分依赖安装失败，请查看上方日志" }
else { Ok "依赖安装完成" }

# ---------- 4. 自检 ----------
Info "4/4 运行环境自检"
& $PY "$REPO\tools\check_env.py"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "环境路径: $PY" -ForegroundColor White
Write-Host "在 PyCharm 中: Settings -> Python Interpreter -> 选择该路径" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
