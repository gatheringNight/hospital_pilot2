# 本地模型文件

- 文件：`models/qwen2.5-3b-instruct-q4_k_m.gguf`
- 来源：用户已下载的本地 GGUF 权重；其 SHA-256 与 [Qwen 官方 GGUF 仓库中的同名文件](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/qwen2.5-3b-instruct-q4_k_m.gguf) 一致。本机原始下载位置不是院内部署路径。
- 文件大小：2,104,932,768 字节。
- SHA-256：`626B4A6678B86442240E33DF819E00132D3BA7DDDFE1CDC4FBB18E0A9615C62D`；源文件与包内文件一致。
- 推理服务：需在院内预装兼容 GGUF 的 `llama-server`。`start_local_model.ps1` 和 `start_local_model.sh` 由用户手动以前台方式启动，只监听 `127.0.0.1`。
- 仓库配置：完整 GGUF 留在本机且被 Git 忽略；`tools/model_parts.py split` 可生成每片不超过 1,000,000,000 字节的 Git LFS 分片和哈希清单。克隆后启动脚本会调用 `restore`，验证分片与完整模型的 SHA-256 后再启动。院内若通过批准的离线介质提供完整权重，也能直接通过相同校验。
- 许可：仓库已附 `models/QWEN_LICENSE.txt` 和 `models/NOTICE.txt`。官方 [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/LICENSE) 允许按其条款为非商业研究或评估使用及再分发；商业用途需要另行取得许可。Git LFS 存储与下载额度仍受仓库账户设置约束。
