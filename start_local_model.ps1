param(
    [string]$LlamaServer = "llama-server.exe",
    [int]$GpuLayers = 0,
    [int]$Port = 18080
)

$modelPath = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct-q4_k_m.gguf"
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "缺少本地 GGUF 模型：$modelPath"
}
& $LlamaServer -m $modelPath --host 127.0.0.1 --port $Port -ngl $GpuLayers -c 4096 --no-webui
if ($LASTEXITCODE -ne 0) {
    throw "llama-server 退出，代码：$LASTEXITCODE"
}
