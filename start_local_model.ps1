param(
    [string]$LlamaServer = "llama-server.exe",
    [string]$Python = "python",
    [int]$GpuLayers = 0,
    [int]$Port = 18080
)

$modelPath = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct-q4_k_m.gguf"
& $Python (Join-Path $PSScriptRoot "tools\model_parts.py") restore
if ($LASTEXITCODE -ne 0) {
    throw "模型还原或SHA-256校验失败；检查models分片和Git LFS下载"
}
& $LlamaServer -m $modelPath --host 127.0.0.1 --port $Port -ngl $GpuLayers -c 4096 --no-webui
if ($LASTEXITCODE -ne 0) {
    throw "llama-server 退出，代码：$LASTEXITCODE"
}
