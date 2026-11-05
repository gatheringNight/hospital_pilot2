# 院内离线流程：Q/GT/K、KB/KG、RAG 与 Agent1—Agent3

本包在**院内服务器**运行。先进入部署后的 `hospital_pilot2` 目录；下文命令均使用相对路径，`python` 和 `llama-server` 指院内安装的程序。完整流程边界见 [FROZEN_PIPELINE.md](FROZEN_PIPELINE.md)。

开始①之前，按以下顺序准备数据：

1. 将院内获准使用的原始分表导出为**带表头的 UTF-8 CSV**，放入 `private/exports/`。每张表一个文件。文件名改为老师所给表组名加 `.csv`；`config/field_schema_manifest.json` 已按《医院各表字段.xlsx》记录16组、244个字段位置，是程序规则文件，不是病例数据文件。
2. **必须有 `就诊信息.csv`**，其表头至少包含 `EMPIID`、`就诊流水号`、`入院日期`。为生成医学证据任务，还须有**至少一张可与就诊表关联的临床表**，例如 `护理信息.csv`、`检验信息.csv`、`诊断信息.csv` 或 `药品信息.csv`。提供的临床表须包含 `EMPIID`、`就诊流水号`，且同一患者、同一次就诊下有足够的非空临床记录；只提供就诊表通常会得到0道题。其他表均可选。
3. 确认各表的 `EMPIID` 和 `就诊流水号` 值能够对应；`入院日期` 使用 `YYYY-MM-DD` 或 `YYYY/MM/DD`。在①执行01后，先检查 `private/step1_output/prepare_summary.json` 中的 `tasks` 是否大于0，并查看字段覆盖及被排除的病例，再按下文命令进入02。

默认直接采用老师提供的字段名。**首次院内导出的主要风险**是 CSV 内部列名与给出的字段名不完全一致；只改文件名不能解决列名不一致。遇到这种情况，再用 `config/table_mapping.example.json` 的 `field_aliases`、`files` 配置实际名称，并以 `--config` 指定映射文件。原始病例、患者对照和生成结果均保存在 Git 忽略的 `private/`。

## ① 准备任务，生成 Q/GT/K

在院内环境安装依赖，准备病例证据卡和问题任务：

```shell
python -m pip install -r requirements-local.txt
python step1/01_prepare_qgtk_tasks.py --resume
```

若院内服务器不能联网，须提前准备 `requirements-local.txt` 对应的离线 Python 安装包。01结束后查看 `private/step1_output/prepare_summary.json`，确认 `tasks > 0` 再运行02。

Linux 还需准备一次 `llama-server`：本仓库不附带操作系统相关的可执行程序。由院方按规定取得 [llama.cpp 官方源码](https://github.com/ggml-org/llama.cpp)，放在本仓库同级的 `../llama.cpp`；允许联网的环境可在本仓库目录运行 `git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp`。院内服务器需有 CMake 和 C++ 编译器。在 `hospital_pilot2` 目录执行 CPU 版构建：

```bash
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release --target llama-server -j 4
```

构建后可用 `../llama.cpp/build/bin/llama-server --version` 检查。若院内已有兼容的 `llama-server`，可跳过构建。无外网时须提前通过院方批准的方式准备源码或兼容的 Linux 可执行程序；不要在院内直接运行需要联网的下载命令。

先确认系统已安装 Git LFS；`git lfs install` 只负责初始化 Git LFS，不会安装这个程序。再补齐模型分片。启动脚本会自动合并分片、校验 SHA-256；也可以用第三条命令提前手动检查还原。已有完整模型时只校验，不重复合并。无需手工拼接分片。

```shell
git lfs install
git lfs pull
python tools/model_parts.py restore
```

另开终端，手动启动本地模型。Windows PowerShell 使用第一条；Linux 若按上面的同级目录构建，使用第二条。Linux 已将 `llama-server` 放入 `PATH` 时也可直接运行 `bash start_local_model.sh`。CPU 使用 0 层 GPU 卸载；需要 GPU 时再按院内硬件调整。

```powershell
.\start_local_model.ps1 -GpuLayers 0
```

```bash
LLAMA_SERVER="$PWD/../llama.cpp/build/bin/llama-server" bash start_local_model.sh
```

保持模型服务运行，在另一个终端生成并复核 Q/GT/K：

```shell
python step1/02_run_local_qgtk.py --stage run --resume
python step1/02_run_local_qgtk.py --stage audit
```

首次运行01后还要查看 `private/step1_output/field_coverage_244.json` 和 `excluded_cases.jsonl`，确认关联情况。01输出 `private/step1_output/qgtk_tasks.jsonl`；02输出同目录的 `qgtk_results.jsonl` 和 `qgtk_summary.json`。02审核后确认 `passed > 0`、检查 `rejected` 与 `pending`，再进入②；`passed` 仅表示自动来源和值检查通过，文字仍需院内抽查。

## ② 生成相近的错误答案

先选定每题要改动的一个来源事实，再由已启动的本地模型生成完整答案，最后离线复核：

```shell
python step2/03_generate_local_near_truth_answers.py --stage prepare
python step2/03_generate_local_near_truth_answers.py --stage run --resume
python step2/03_generate_local_near_truth_answers.py --stage audit
```

结果在 `private/step2_output/poison_answer_results.jsonl`。检查同目录的 `summary.json`，确认 `passed > 0` 并了解 `rejected`、`pending`；③只使用通过的配对。构造时的预设事实差异不等于后续 Agent2 的检出结果。

## ③ 建立 KB/KG

此步不用模型服务。每道通过配对的题目建立一篇 clean 文档和一篇 poison 文档；可信 KG 使用经原始行校验且写入 Q/GT/K 的证据卡，不读取错误答案。

```shell
python step3/04_build_local_kb_kg.py --stage build
python step3/04_build_local_kb_kg.py --stage audit
```

输出在 `private/step3_output/`：`kb_documents.jsonl` 是文档对，`kb.sqlite` 和两个 TF-IDF 目录是检索库，`trusted_kg_facts.jsonl`、`trusted_kg.sqlite`、`trusted_kg.graphml` 是可信事实图，`target_coverage.jsonl` 对应每题错误目标的可信事实；`build_summary.json` 记录数量。院内运行后再检查实际覆盖与文档质量。

## ④ 模拟用户提问与 RAG 回答

01生成的 `Question` 充当用户问题。先离线冻结按病例范围检索的 Top-1 计划；再手动保持本地模型服务运行，生成 clean 参照回答与混合 KB 自然回答；最后离线汇总：

```shell
python step4/05_run_local_rag_answers.py --stage prepare
python step4/05_run_local_rag_answers.py --stage run --resume
python step4/05_run_local_rag_answers.py --stage audit
```

结果在 `private/step4_output/`。`rag_results.jsonl` 保留两条线和检索来源，`agent1_natural_top1_inputs.jsonl` 是下一步 Agent1 的自然线输入，`agent1_clean_only_inputs.jsonl` 是参照线输入；模型生成时只见问题和文档正文。检查 `summary.json` 的 `pending_answers` 和各场景 `rejected`；Agent3 要求自然线回答完整通过。`natural_top1_poison_retrievals` 仅是毒文被检索的次数，不是攻击成功次数。

## ⑤ Agent1：拆分原子 claim

默认读取自然线回答。先冻结输入任务，再用已启动的院内本地模型抽取可单独核查的事实，最后离线汇总：

```shell
python step5/06_extract_local_atomic_claims.py --stage prepare
python step5/06_extract_local_atomic_claims.py --stage run --resume
python step5/06_extract_local_atomic_claims.py --stage audit
```

输出在 `private/step5_agent1_natural/`：`agent1_atomic_claims.jsonl` 是下一步 Agent2 的逐条输入，`agent1_results.jsonl` 保留逐回答结果和重试记录。Agent1 只看 Question 与 RAG Answer；来源原句从回答中回填，**此处不判定 support 或 contradiction**。

## ⑥ Agent2：可信 KG 事实核查

先从同病例可信 KG 为每条 claim 冻结候选证据，再由院内本地模型逐条判定，最后离线汇总：

```shell
python step6/07_check_local_claims_with_kg.py --stage prepare
python step6/07_check_local_claims_with_kg.py --stage run --resume
python step6/07_check_local_claims_with_kg.py --stage audit
```

结果在 `private/step6_agent2_natural/`：`agent2_claim_labels.jsonl` 保存 support、contradiction、unverifiable 及引用的 KG 事实，供下一步 Agent2A 使用；`agent2_results.jsonl` 留存逐次模型回复。此步不读取 KB 毒文、GT 或目标错误标签，预测标签不会被预设标签覆盖。

## ⑦ Agent2A：检索来源归因

只路由 Agent2 判为 `contradiction` 且引用可信 KG 的 claim。先离线核对回答、实际检索的 KB 文档和 KG 原始行来源，再由院内本地模型判断检索文本与 claim 的字面关系，最后汇总：

```shell
python step7/08_attribute_local_sources.py --stage prepare
python step7/08_attribute_local_sources.py --stage run --resume
python step7/08_attribute_local_sources.py --stage audit
```

输出在 `private/step7_agent2a_natural/`：`source_attributions.jsonl` 留存逐条文本对齐和 KG 来源，`evidence_chains_by_answer.jsonl` 按回答汇总。文档角色在对齐后才附入；来源对齐说明文本关联，**不单独证明该文档造成了错误**。若 Agent2 未检出 contradiction，计划和汇总会如实显示 0 条。

## ⑧ Agent2B：删除、增加、替换反事实

只对 Agent2A 已追溯到毒文的 claim 做四组干预：原文控制、删除原文、原文加同题 clean 文档、用同题 clean 文档替换。先冻结任务，再用院内本地模型生成回答，**生成完成后**单独核对各回答，最后离线汇总：

```shell
python step8/09_verify_local_counterfactuals.py --stage prepare
python step8/09_verify_local_counterfactuals.py --stage run --resume
python step8/09_verify_local_counterfactuals.py --stage judge --resume
python step8/09_verify_local_counterfactuals.py --stage audit
```

输出在 `private/step8_agent2b_natural/`：`counterfactual_responses.jsonl` 是新生成的四组回答，`counterfactual_judgments.jsonl` 是生成后的逐 claim 核对，`counterfactual_effects_by_claim.jsonl` 记录控制组是否复现、删除／增加／替换的变化。判断仍由同一院内本地模型完成，须人工抽查；控制组未复现时不把后续变化解释为来源依赖。

## ⑨ Agent3：离线聚合与证据链

确认前面各步均完成后运行。此步不调用模型，按回答汇总 Agent2 事实标签、Agent2A 来源链及 Agent2B 四组变化；缺失结果会明确报错。

```shell
python step9/10_aggregate_local_agent_evidence.py
```

输出在 `private/step9_agent3_natural/`：`agent3_judgments.jsonl` 是逐回答结论，`agent3_evidence_graphs.jsonl` 是回答→claim→可信KG／检索来源／反事实的链，`agent3_review_queue.jsonl` 列出需人工复核的项目，`summary.json` 汇总数量。事实冲突与投毒来源分开报告；本步不读取正确答案或预设目标错误，也不计算最终评价指标。

## 模型分片与自动合并

仓库用 Git LFS 保存三个 `.partXX` 分片及 `models/model_parts.json` 清单；完整 `.gguf` 被 `.gitignore` 排除。克隆后运行上方的 `git lfs pull`，再执行 `python tools/model_parts.py restore`，或直接运行 `start_local_model.ps1` / `start_local_model.sh` 自动还原。脚本逐片和整体验证 SHA-256；还原后的完整模型仅留在本机。需预留约 2.1 GB 空间存放还原文件。院内无法访问 GitHub 时，须通过院方批准的离线方式提供已下载的仓库和分片。

模型来源、许可证与声明见 [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md)、`models/QWEN_LICENSE.txt` 和 `models/NOTICE.txt`。模型许可限于非商业研究或评估用途，其他用途须按许可证取得相应授权。
