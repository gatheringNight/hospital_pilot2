"""Agent2B: frozen four-way local RAG context interventions and later claim review.

Generation reuses step4's exact prompt and model call. Only Question and assigned
document texts are visible to that model. Post-generation judgments are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPT = ROOT / "step4/05_run_local_rag_answers.py"
SOURCE = ROOT / "private/step7_agent2a_natural/source_attributions.jsonl"
RAG_RESULTS = ROOT / "private/step4_output/rag_results.jsonl"
RAG_PLAN = ROOT / "private/step4_output/rag_plan.jsonl"
KB = ROOT / "private/step3_output/kb_documents.jsonl"
OUTPUT = ROOT / "private/step8_agent2b_natural"
LOCAL_URL = "http://127.0.0.1:18080/v1/chat/completions"
SCHEMA = "hospital-local-agent2b-counterfactual-v1"
VARIANTS = ("CONTROL_SOURCE", "DELETE_SOURCE", "ADD_TRUSTED", "REPLACE_TRUSTED")
STATES = ("present", "absent", "unclear")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes(source: Path, rag_results: Path, rag_plan: Path, kb: Path) -> dict:
    return {"agent2a": file_hash(source), "rag_results": file_hash(rag_results),
            "rag_plan": file_hash(rag_plan), "kb_documents": file_hash(kb),
            "rag_script": file_hash(RAG_SCRIPT)}


def unique(rows: list[dict], key: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        item_id = row.get(key)
        if not item_id or item_id in indexed:
            raise ValueError(f"{key}为空或重复：{item_id}")
        indexed[item_id] = row
    return indexed


def load_rag_module():
    spec = importlib.util.spec_from_file_location("hospital_pilot2_frozen_rag", RAG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("找不到冻结的RAG生成脚本")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def doc_view(doc: dict) -> dict:
    if doc.get("text_sha256") != text_hash(doc["text"]):
        raise ValueError(f"KB文档哈希不符：{doc['doc_id']}")
    return {"doc_id": doc["doc_id"], "doc_role": doc["doc_role"],
            "text": doc["text"], "text_sha256": doc["text_sha256"]}


def generation_messages(task: dict, rag_module) -> list[dict]:
    # Never pass task metadata, role, KG citations, or the original answer.
    return rag_module.model_messages({"Question": task["Question"],
                                      "retrieved": [{"text": doc["text"]}
                                                    for doc in task["documents"]]})


def make_plan(source_path: Path, rag_path: Path, rag_plan_path: Path,
              kb_path: Path, rag_module) -> tuple[list[dict], list[dict], dict]:
    sources = read_jsonl(source_path)
    rag = unique(read_jsonl(rag_path), "answer_id")
    original_plans = unique(read_jsonl(rag_plan_path), "answer_id")
    documents = unique(read_jsonl(kb_path), "doc_id")
    pair_index = defaultdict(list)
    for doc in documents.values():
        pair_index[(doc["case_id"], doc["question_id"], doc["doc_role"])].append(doc)
    selected = [row for row in sources if row.get("status") == "passed" and
                row.get("complete_source_chain") and row.get("poisoned_source_chain") and
                (row.get("claim_to_retrieved_source", {}).get("alignment") or {}).get("relation") == "entails"]
    claims, generations_by_answer, seen = [], {}, set()
    for row in selected:
        claim_id, answer_id = row["claim_id"], row["answer_id"]
        if claim_id in seen:
            raise ValueError(f"重复的Agent2B claim：{claim_id}")
        seen.add(claim_id)
        original = rag.get(answer_id)
        original_plan = original_plans.get(answer_id)
        alignment = row["claim_to_retrieved_source"]
        suspect = documents.get(alignment["doc_id"])
        if (not original or not original_plan or not suspect or
                original.get("status") != "passed" or original["scenario"] != "natural_top1" or
                any(original.get(key) != row.get(key) for key in ("case_id", "question_id", "scenario")) or
                original["Question"] != original_plan["Question"] or
                original["RAG Answer"] != row["answer_to_claim"]["RAG Answer"] or
                row["answer_to_claim"]["answer_quote"] not in original["RAG Answer"] or
                len(original["retrieved"]) != 1 or
                original["retrieved"][0]["doc_id"] != suspect["doc_id"] or
                original_plan["retrieved"] != original["retrieved"] or
                suspect["case_id"] != row["case_id"] or suspect["doc_role"] != "poisoned_doc" or
                original["retrieved"][0]["text"] != suspect["text"] or
                text_hash(suspect["text"]) != alignment["text_sha256"]):
            raise ValueError(f"冻结RAG、Agent2A与KB来源不一致：{claim_id}")
        clean_matches = pair_index[(suspect["case_id"], suspect["question_id"], "clean_doc")]
        if len(clean_matches) != 1:
            raise ValueError(f"检索毒文缺少唯一同题clean配对：{claim_id}")
        clean = clean_matches[0]
        if clean["doc_id"] == suspect["doc_id"]:
            raise ValueError("clean/poison配对文档ID相同")
        trusted = [item["statement"] for item in row["claim_to_trusted_kg"]["kg_counterevidence"]]
        if not trusted or row["claim_to_trusted_kg"]["agent2_label"] != "contradiction":
            raise ValueError(f"缺少冻结的KG反证：{claim_id}")
        claim = {"schema_version": SCHEMA, "claim_id": claim_id,
                 "answer_id": answer_id, "question_id": row["question_id"],
                 "case_id": row["case_id"], "scenario": row["scenario"],
                 "claim_text": row["answer_to_claim"]["claim_text"],
                 "trusted_statements": trusted,
                 "frozen_original_answer": original["RAG Answer"],
                 "frozen_original_answer_sha256": text_hash(original["RAG Answer"]),
                 "source_doc_id": suspect["doc_id"],
                 "synthetic_only": row.get("synthetic_only", False)}
        claim["claim_sha256"] = digest(claim)
        claims.append(claim)
        if answer_id in generations_by_answer:
            if generations_by_answer[answer_id][0]["source_doc_id"] != suspect["doc_id"]:
                raise ValueError(f"同一回答的反事实来源不一致：{answer_id}")
            continue
        suspect_view, clean_view = doc_view(suspect), doc_view(clean)
        assignments = {"CONTROL_SOURCE": [suspect_view], "DELETE_SOURCE": [],
                       "ADD_TRUSTED": [suspect_view, clean_view],
                       "REPLACE_TRUSTED": [clean_view]}
        tasks = []
        for variant in VARIANTS:
            task = {"schema_version": SCHEMA, "answer_id": answer_id,
                    "question_id": row["question_id"], "case_id": row["case_id"],
                    "scenario": row["scenario"], "variant": variant,
                    "Question": original["Question"], "documents": assignments[variant],
                    "source_doc_id": suspect["doc_id"],
                    "clean_doc_id": clean["doc_id"],
                    "frozen_original_answer_sha256": text_hash(original["RAG Answer"]),
                    "synthetic_only": row.get("synthetic_only", False)}
            task["generation_id"] = digest([answer_id, variant])[:24]
            task["task_sha256"] = digest(task)
            tasks.append(task)
            if variant == "CONTROL_SOURCE" and generation_messages(task, rag_module) != (
                    rag_module.model_messages(original_plan)):
                raise ValueError(f"控制组与冻结RAG提示词不一致：{answer_id}")
        generations_by_answer[answer_id] = tasks
    generations = [task for answer_id in sorted(generations_by_answer)
                   for task in generations_by_answer[answer_id]]
    report = {"agent2a_attributions": len(sources), "eligible_poisoned_claims": len(claims),
              "distinct_answers": len(generations_by_answer),
              "planned_generations": len(generations),
              "variants": list(VARIANTS), "control_prompt_matches_frozen_rag": True,
              "generation_input_fields": ["Question", "documents.text"],
              "generation_hidden_fields": ["doc_role", "KG", "error_claim", "original_answer"]}
    return claims, generations, report


def valid_generation(task: dict, row: dict, rag_module) -> bool:
    return (row.get("status") == "passed" and row.get("task_sha256") == task["task_sha256"] and
            row.get("generation_id") == task["generation_id"] and
            isinstance(row.get("generated_answer"), str) and
            not rag_module.answer_issues(row["generated_answer"], task))


def generate(task: dict, rag_module, url: str, timeout: int, retries: int) -> dict:
    attempts, answer = [], None
    for number in range(1, retries + 1):
        raw = None
        try:
            raw = rag_module.local_call(url, generation_messages(task, rag_module), timeout)
            candidate = rag_module.parse_answer(raw)
            issues = rag_module.answer_issues(candidate, task)
            attempts.append({"attempt": number, "raw_response": raw, "issues": issues})
            if not issues:
                answer = candidate
                break
        except Exception as error:
            attempts.append({"attempt": number, "raw_response": raw, "error": str(error)})
    return {"schema_version": SCHEMA, "generation_id": task["generation_id"],
            "answer_id": task["answer_id"], "question_id": task["question_id"],
            "variant": task["variant"], "task_sha256": task["task_sha256"],
            "status": "passed" if answer is not None else "rejected",
            "generated_answer": answer, "attempts": attempts,
            "model_input_fields": ["Question", "documents.text"],
            "synthetic_only": task["synthetic_only"]}


def judge_messages(claim: dict, generated_answer: str, feedback: str | None = None) -> list[dict]:
    system = (
        "你是反事实回答的单独文本核对员。只依据给出的生成回答判断其是否明确表达测试claim，"
        "以及是否完整表达给出的可信KG陈述；不要凭医学常识改写回答。"
        "每项只用present、absent、unclear。若回答含糊或只支持部分内容，请用unclear。"
        "这是回答生成之后的分析，不要继续回答用户问题。"
        "只返回JSON：{\"error_claim_status\":\"present|absent|unclear\","
        "\"trusted_fact_status\":\"present|absent|unclear\",\"rationale\":\"简短理由\"}。"
    )
    payload = {"generated_answer": generated_answer, "tested_claim": claim["claim_text"],
               "trusted_kg_statements": claim["trusted_statements"]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if feedback:
        messages.append({"role": "user", "content": "上次返回格式有误：" + feedback[:240] +
                         "。只返回指定JSON，不补充新事实。"})
    return messages


def local_judge_call(url: str, messages: list[dict], timeout: int) -> str:
    target = urllib.parse.urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1"} or not target.port:
        raise ValueError("Agent2B只允许连接本机模型服务")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            raise ValueError("本机模型接口重定向已拒绝")

    body = json.dumps({"model": "local-qwen", "messages": messages,
                       "temperature": 0.0, "max_tokens": 500,
                       "response_format": {"type": "json_object"}},
                      ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def parse_json(raw: str) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I).strip()
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            raise ValueError("核对模型没有返回JSON对象")
        obj, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(obj, dict):
        raise ValueError("核对模型输出不是JSON对象")
    return obj


def validate_judgment(obj: dict) -> dict:
    error = str(obj.get("error_claim_status") or "").strip().casefold()
    trusted = str(obj.get("trusted_fact_status") or "").strip().casefold()
    if error not in STATES or trusted not in STATES:
        raise ValueError("反事实核对状态无效")
    return {"error_claim_status": error, "trusted_fact_status": trusted,
            "rationale": str(obj.get("rationale") or "").strip()[:500]}


def judge(claim: dict, task: dict, generation: dict,
          url: str, timeout: int, retries: int) -> dict:
    answer = generation["generated_answer"]
    attempts, judgment, feedback = [], None, None
    for number in range(1, retries + 1):
        raw = None
        try:
            raw = local_judge_call(url, judge_messages(claim, answer, feedback), timeout)
            judgment = validate_judgment(parse_json(raw))
            attempts.append({"attempt": number, "raw_response": raw, "error": None})
            break
        except Exception as error:
            feedback = str(error)
            attempts.append({"attempt": number, "raw_response": raw, "error": feedback})
    return {"schema_version": SCHEMA, "claim_id": claim["claim_id"],
            "answer_id": claim["answer_id"], "variant": task["variant"],
            "generation_id": task["generation_id"],
            "claim_sha256": claim["claim_sha256"],
            "generated_answer_sha256": text_hash(answer),
            "status": "passed" if judgment is not None else "rejected",
            "judgment": judgment, "attempts": attempts,
            "post_generation_only": True,
            "requires_manual_review": True}


def valid_judgment(row: dict, claim: dict, task: dict, generation: dict) -> bool:
    if (row.get("status") != "passed" or row.get("claim_sha256") != claim["claim_sha256"] or
            row.get("generation_id") != task["generation_id"] or
            row.get("generated_answer_sha256") != text_hash(generation["generated_answer"])):
        return False
    try:
        return validate_judgment(row["judgment"]) == row["judgment"]
    except (KeyError, TypeError, ValueError):
        return False


def collect_generations(tasks: list[dict], output_dir: Path) -> dict[str, dict]:
    saved = {}
    for task in tasks:
        path = output_dir / "generation_work" / f"{task['generation_id']}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("task_sha256") != task["task_sha256"]:
                raise ValueError(f"{task['generation_id']} 的反事实生成任务已变化")
            saved[task["generation_id"]] = row
    return saved


def collect_judgments(claims: list[dict], tasks: list[dict], output_dir: Path) -> dict[tuple[str, str], dict]:
    saved = {}
    by_answer_variant = {(task["answer_id"], task["variant"]): task for task in tasks}
    for claim in claims:
        for variant in VARIANTS:
            task = by_answer_variant[(claim["answer_id"], variant)]
            judgment_id = digest([claim["claim_id"], variant])[:24]
            path = output_dir / "judgment_work" / f"{judgment_id}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row.get("claim_sha256") != claim["claim_sha256"] or
                        row.get("generation_id") != task["generation_id"]):
                    raise ValueError(f"{claim['claim_id']}/{variant} 的核对任务已变化")
                saved[(claim["claim_id"], variant)] = row
    return saved


def summarize(claims: list[dict], tasks: list[dict], generations: dict[str, dict],
              judgments: dict[tuple[str, str], dict], output_dir: Path, rag_module) -> dict:
    generation_rows = [generations[task["generation_id"]] for task in tasks
                       if task["generation_id"] in generations]
    judgment_rows = [judgments[(claim["claim_id"], variant)] for claim in claims
                     for variant in VARIANTS if (claim["claim_id"], variant) in judgments]
    write_jsonl(output_dir / "counterfactual_responses.jsonl", generation_rows)
    write_jsonl(output_dir / "counterfactual_judgments.jsonl", judgment_rows)
    by_answer_variant = {(task["answer_id"], task["variant"]): task for task in tasks}
    effects = []
    for claim in claims:
        states = {}
        for variant in VARIANTS:
            task = by_answer_variant[(claim["answer_id"], variant)]
            generation = generations.get(task["generation_id"], {})
            row = judgments.get((claim["claim_id"], variant), {})
            if not valid_generation(task, generation, rag_module) or not valid_judgment(
                    row, claim, task, generation):
                continue
            states[variant] = row["judgment"]
        if len(states) != len(VARIANTS):
            continue
        control = states["CONTROL_SOURCE"]["error_claim_status"] == "present"
        deletion_removed = control and states["DELETE_SOURCE"]["error_claim_status"] == "absent"
        addition_corrected = (control and states["ADD_TRUSTED"]["error_claim_status"] == "absent" and
                              states["ADD_TRUSTED"]["trusted_fact_status"] == "present")
        replacement_corrected = (control and states["REPLACE_TRUSTED"]["error_claim_status"] == "absent" and
                                 states["REPLACE_TRUSTED"]["trusted_fact_status"] == "present")
        effects.append({"claim_id": claim["claim_id"], "answer_id": claim["answer_id"],
                        "states": states, "control_reproduced": control,
                        "deletion_error_removed_given_control": deletion_removed,
                        "addition_correction_given_control": addition_corrected,
                        "replacement_correction_given_control": replacement_corrected,
                        "frozen_original_answer_sha256": claim["frozen_original_answer_sha256"],
                        "source_dependence_observed": deletion_removed,
                        "not_formal_causal_proof": True,
                        "judgments_require_manual_review": True})
    write_jsonl(output_dir / "counterfactual_effects_by_claim.jsonl", effects)
    valid_judgment_count = 0
    for claim in claims:
        for variant in VARIANTS:
            task = by_answer_variant[(claim["answer_id"], variant)]
            generation = generations.get(task["generation_id"], {})
            row = judgments.get((claim["claim_id"], variant), {})
            if valid_generation(task, generation, rag_module) and valid_judgment(
                    row, claim, task, generation):
                valid_judgment_count += 1
    summary = {"schema_version": SCHEMA, "eligible_claims": len(claims),
               "distinct_answers": len(tasks) // len(VARIANTS),
               "planned_generations": len(tasks),
               "passed_generations": sum(valid_generation(task, generations.get(task["generation_id"], {}),
                                           rag_module) for task in tasks),
               "pending_generations": len(tasks) - len(generation_rows),
               "planned_judgments": len(claims) * len(VARIANTS),
               "passed_judgments": valid_judgment_count,
               "pending_judgments": len(claims) * len(VARIANTS) - len(judgment_rows),
               "complete_claims": len(effects),
               "control_reproduced": sum(row["control_reproduced"] for row in effects),
               "deletion_error_removed_given_control": sum(
                   row["deletion_error_removed_given_control"] for row in effects),
               "addition_correction_given_control": sum(
                   row["addition_correction_given_control"] for row in effects),
               "replacement_correction_given_control": sum(
                   row["replacement_correction_given_control"] for row in effects),
               "frozen_original_answers_reused_as_generated": False,
               "generation_prompt_reused_from_stage4": True,
               "generation_sees_role_or_kg": False,
               "judge_is_post_generation": True,
               "judge_uses_same_local_model_family": True,
               "not_formal_causal_proof": True,
               "requires_manual_review": True}
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "judge", "audit"), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--rag", type=Path, default=RAG_RESULTS)
    parser.add_argument("--rag-plan", type=Path, default=RAG_PLAN)
    parser.add_argument("--kb-documents", type=Path, default=KB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=0,
                        help="0处理全部；正数仅供院内小批预检")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_items < 0 or args.max_retries < 1:
        parser.error("max-items不能为负数，max-retries至少为1")
    rag_module = load_rag_module()
    claim_path = args.output_dir / "counterfactual_claims.jsonl"
    task_path = args.output_dir / "counterfactual_tasks.jsonl"
    manifest_path = args.output_dir / "plan_manifest.json"
    if args.stage == "prepare":
        if claim_path.exists() or task_path.exists():
            raise FileExistsError("Agent2B计划已冻结；请继续run或使用新输出目录")
        claims, tasks, report = make_plan(args.source, args.rag, args.rag_plan,
                                          args.kb_documents, rag_module)
        write_jsonl(claim_path, claims)
        write_jsonl(task_path, tasks)
        write_json(manifest_path, {"schema_version": SCHEMA,
                                   "source_hashes": source_hashes(
                                       args.source, args.rag, args.rag_plan, args.kb_documents), **report})
        print(json.dumps(report, ensure_ascii=False))
        return
    if not claim_path.is_file() or not task_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("先执行 --stage prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_hashes"] != source_hashes(args.source, args.rag, args.rag_plan, args.kb_documents):
        raise ValueError("Agent2A、RAG、KB或RAG脚本已变化；不能混用冻结的反事实任务")
    claims, tasks = read_jsonl(claim_path), read_jsonl(task_path)
    generations = collect_generations(tasks, args.output_dir)
    judgments = collect_judgments(claims, tasks, args.output_dir)
    if args.stage == "run":
        if generations and not args.resume:
            raise ValueError("已有反事实回答；继续生成请加 --resume")
        for task in tasks[:args.max_items or None]:
            old = generations.get(task["generation_id"], {})
            if valid_generation(task, old, rag_module):
                continue
            row = generate(task, rag_module, args.local_url, args.timeout, args.max_retries)
            write_json(args.output_dir / "generation_work" / f"{task['generation_id']}.json", row)
            generations[task["generation_id"]] = row
            print(f"{task['answer_id']} {task['variant']}: {row['status']}", flush=True)
    elif args.stage == "judge":
        if judgments and not args.resume:
            raise ValueError("已有反事实核对结果；继续核对请加 --resume")
        by_answer_variant = {(task["answer_id"], task["variant"]): task for task in tasks}
        selected = [(claim, variant) for claim in claims for variant in VARIANTS]
        for claim, variant in selected[:args.max_items or None]:
            task = by_answer_variant[(claim["answer_id"], variant)]
            generation = generations.get(task["generation_id"], {})
            if not valid_generation(task, generation, rag_module):
                continue
            old = judgments.get((claim["claim_id"], variant), {})
            if valid_judgment(old, claim, task, generation):
                continue
            row = judge(claim, task, generation, args.local_url, args.timeout, args.max_retries)
            judgment_id = digest([claim["claim_id"], variant])[:24]
            write_json(args.output_dir / "judgment_work" / f"{judgment_id}.json", row)
            judgments[(claim["claim_id"], variant)] = row
            print(f"{claim['claim_id']} {variant}: {row['status']}", flush=True)
    print(json.dumps(summarize(claims, tasks, generations, judgments, args.output_dir, rag_module),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
