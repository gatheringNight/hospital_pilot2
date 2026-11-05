# 本地模型文件

- 文件：`models/qwen2.5-3b-instruct-q4_k_m.gguf`
- 本轮来源：用户已下载的本地 GGUF 权重；整合包时逐字节复制。本机原始下载位置不是院内部署路径。
- 文件大小：2,104,932,768 字节。
- SHA-256：`626B4A6678B86442240E33DF819E00132D3BA7DDDFE1CDC4FBB18E0A9615C62D`；源文件与包内文件一致。
- 推理服务：需在院内预装兼容 GGUF 的 `llama-server`。`start_local_model.ps1` 和 `start_local_model.sh` 由用户手动以前台方式启动，只监听 `127.0.0.1`。
- 仓库配置：`.gitattributes` 已将 GGUF 指定为 Git LFS 文件。发布 GitHub 前需核对原下载页的模型许可证、GGUF 的再分发许可和目标仓库的 LFS 存储条件；当前本机目录没有这些许可文件，故尚未宣布可以公开发布权重。院内若通过批准的离线介质提供权重，也可保持同一路径而不公开上传。
