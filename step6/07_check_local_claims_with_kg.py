"""Check Agent1 atomic claims against the isolated trusted hospital KG.

The local model sees one claim, its question, and patient-scoped KG statements.
KB documents, gold answers, document roles, and poison targets are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "private/step5_agent1_natural/agent1_atomic_claims.jsonl"
KG_DIR = ROOT / "private/step3_output"
OUTPUT = ROOT / "private/step6_agent2_natural"
LOCAL_URL = "http://127.0.0.1:18080/v1/chat/completions"
SCHEMA = "hospital-local-agent2-kg-check-v1"
LABELS = ("support", "contradiction", "unverifiable")
ALIASES = {"支持": "support", "一致": "support", "supported": "support",
           "矛盾": "contradiction", "冲突": "contradiction", "不一致": "contradiction",
           "contradictory": "contradiction", "无法核实": "unverifiable",
           "不可验证": "unverifiable", "无法判断": "unverifiable", "不确定": "unverifiable"}
DIRECTIONS = {"decreased": "下降", "increased": "上升", "unchanged": "不变"}
TREND_WORDS = ("上升", "增加", "升高", "下降", "降低", "减少", "不变", "稳定", "变化")


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


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes(claims_path: Path, kg_dir: Path) -> dict:
    return {"agent1_claims": file_hash(claims_path),
            "trusted_kg_facts": file_hash(kg_dir / "trusted_kg_facts.jsonl"),
            "trusted_kg_changes": file_hash(kg_dir / "trusted_kg_changes.jsonl")}


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def bigrams(value: object) -> set[str]:
    text = compact(value)
    return {text[index:index + 2] for index in range(max(0, len(text) - 1))}


def relevance(claim: dict, item: dict) -> float:
    context = compact(" ".join((claim["claim_text"], claim["source_sentence"],
                                claim.get("Question", ""))))
    label = compact(item["label"])
    date = compact(item.get("record_date", ""))
    score = 3.0 if label and label in context else 0.0
    score += 2.0 if date and date in context else 0.0
    score += 2.0 if claim["question_id"] in item.get("question_ids", []) else 0.0
    score += 0.4 if compact(item.get("object", "")) in context else 0.0
    words = bigrams(item["label"])
    if words:
        score += len(words & bigrams(context)) / len(words)
    return round(score, 6)


def fact_statement(fact: dict) -> str:
    return f"{fact['record_date']}记录：{fact['label']}为{fact['object']}。"


def change_statement(change: dict, linked: list[dict]) -> str:
    direction = DIRECTIONS.get(change["object"], change["object"])
    return (f"{change['label']}由{linked[0]['record_date']}的{change['before']}"
            f"变为{linked[1]['record_date']}的{change['after']}，方向为{direction}。")


def candidate_evidence(claim: dict, own_facts: list[dict], own_changes: list[dict],
                       max_candidates: int) -> list[dict]:
    if not own_facts:
        return []
    by_evidence = {fact["evidence_id"]: fact for fact in own_facts}
    candidates = []
    for fact in own_facts:
        candidates.append({"kind": "fact", "kg_id": fact["fact_id"],
                           "case_id": fact["case_id"], "label": fact["label"],
                           "object": fact["object"], "record_date": fact["record_date"],
                           "statement": fact_statement(fact), "question_ids": fact["question_ids"],
                           "source_facts": [{"fact_id": fact["fact_id"], "source": fact["source"]}],
                           "score": relevance(claim, fact)})
    if any(word in claim["claim_text"] for word in TREND_WORDS):
        for change in own_changes:
            linked = [by_evidence[eid] for eid in change["evidence_ids"] if eid in by_evidence]
            if len(linked) != len(change["evidence_ids"]):
                continue
            candidates.append({"kind": "change", "kg_id": change["change_id"],
                               "case_id": change["case_id"], "label": change["label"],
                               "object": change["object"], "record_date": "",
                               "statement": change_statement(change, linked),
                               "question_ids": change["question_ids"],
                               "source_facts": [{"fact_id": fact["fact_id"], "source": fact["source"]}
                                                for fact in linked],
                               "score": relevance(claim, change)})
    candidates.sort(key=lambda item: (-item["score"], item["kg_id"]))
    # The RAG answer may cite another question's document for the same patient.
    # Question linkage is a ranking feature, never a hard evidence filter.
    chosen = candidates[:max_candidates]
    for index, item in enumerate(chosen, 1):
        item["prompt_evidence_id"] = f"K{index}"
    return chosen


def make_plan(claims_path: Path, kg_dir: Path, max_candidates: int) -> tuple[list[dict], dict]:
    claims = read_jsonl(claims_path)
    facts = read_jsonl(kg_dir / "trusted_kg_facts.jsonl")
    changes = read_jsonl(kg_dir / "trusted_kg_changes.jsonl")
    if not claims:
        raise ValueError("Agent1没有可核查的claim")
    if not facts or len({fact["fact_id"] for fact in facts}) != len(facts):
        raise ValueError("可信KG事实为空或ID重复")
    if len({change["change_id"] for change in changes}) != len(changes):
        raise ValueError("可信KG变化事实ID重复")
    facts_by_case, changes_by_case = defaultdict(list), defaultdict(list)
    for fact in facts:
        facts_by_case[fact["case_id"]].append(fact)
    for change in changes:
        changes_by_case[change["case_id"]].append(change)
    seen, scenarios, plan = set(), set(), []
    for claim in claims:
        claim_id = claim.get("claim_id")
        if (not claim_id or claim_id in seen or not claim.get("claim_text") or
                not claim.get("case_id") or not claim.get("source_sentence") or
                not claim.get("question_id") or not claim.get("answer_id")):
            raise ValueError(f"Agent1 claim重复或关键字段缺失：{claim_id}")
        seen.add(claim_id)
        scenarios.add(claim.get("scenario"))
        task = {"schema_version": SCHEMA, "claim_id": claim_id,
                "answer_id": claim["answer_id"], "question_id": claim["question_id"],
                "case_id": claim["case_id"], "scenario": claim["scenario"],
                "Question": claim.get("Question", ""),
                "claim_text": claim["claim_text"],
                "source_sentence": claim["source_sentence"],
                "trusted_kg_evidence": candidate_evidence(
                    claim, facts_by_case[claim["case_id"]],
                    changes_by_case[claim["case_id"]], max_candidates),
                "synthetic_only": claim.get("synthetic_only", False)}
        task["task_sha256"] = digest(task)
        plan.append(task)
    if len(scenarios) != 1 or scenarios.pop() not in {"natural_top1", "clean_only"}:
        raise ValueError("单次Agent2任务必须只含一条回答线")
    counts = Counter(len(task["trusted_kg_evidence"]) for task in plan)
    return plan, {"planned_claims": len(plan),
                  "claims_without_patient_kg": counts[0],
                  "claims_with_question_linked_kg": sum(any(
                      task["question_id"] in item["question_ids"]
                      for item in task["trusted_kg_evidence"]) for task in plan),
                  "candidate_count_distribution": dict(sorted(counts.items())),
                  "model_input_fields": ["Question", "claim_text", "source_sentence",
                                         "trusted_kg_evidence.statement"],
                  "forbidden_inputs": ["KB documents", "Ground Truth", "poison target", "doc_role"]}


def model_messages(task: dict, feedback: str | None = None) -> list[dict]:
    system = (
        "你是院内病例事实核查员。只根据给出的可信KG陈述判断一条claim。"
        "先确认是否指向同一病例中的同一指标、同一时间或就诊，再判断值或事实。"
        "KG明确支持该claim为support；KG对同一事实明确给出相反或不同值为contradiction；"
        "证据不够、指标或时间不能对应时为unverifiable。"
        "不能用回答原句、外部知识或常识当作KG证据。"
        "support和contradiction必须引用提供的K编号；不要编造引用。"
        "只返回JSON：{\"label\":\"support|contradiction|unverifiable\","
        "\"evidence_ids\":[\"K1\"],\"rationale\":\"简短理由\"}。"
    )
    payload = {"Question": task["Question"], "claim": task["claim_text"],
               "answer_source_sentence": task["source_sentence"],
               "trusted_kg_evidence": [{"evidence_id": item["prompt_evidence_id"],
                                        "statement": item["statement"]}
                                       for item in task["trusted_kg_evidence"]]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if feedback:
        messages.append({"role": "user", "content": "上次返回格式或引用有误：" + feedback[:240] +
                         "。请只根据同一事实的KG证据重新判断并输出JSON。"})
    return messages


def local_call(url: str, messages: list[dict], timeout: int) -> str:
    target = urllib.parse.urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1"} or not target.port:
        raise ValueError("Agent2只允许连接本机模型服务")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            raise ValueError("本机模型接口重定向已拒绝")

    body = json.dumps({"model": "local-qwen", "messages": messages,
                       "temperature": 0.0, "max_tokens": 650,
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
            raise ValueError("Agent2没有返回JSON对象")
        obj, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(obj, dict):
        raise ValueError("Agent2输出不是JSON对象")
    return obj


def validate_prediction(task: dict, obj: dict) -> dict:
    label = str(obj.get("label") or "").strip().casefold()
    label = ALIASES.get(label, label)
    if label not in LABELS:
        raise ValueError(f"无效核查标签：{label}")
    ids = obj.get("evidence_ids", obj.get("evidence_id", []))
    if isinstance(ids, str):
        ids = [ids] if ids.strip() else []
    if not isinstance(ids, list):
        raise ValueError("evidence_ids必须是列表")
    ids = [str(item).strip() for item in ids if str(item).strip()]
    if len(ids) > 4 or len(ids) != len(set(ids)):
        raise ValueError("证据引用重复或超过4项")
    offered = {item["prompt_evidence_id"] for item in task["trusted_kg_evidence"]}
    if any(eid not in offered for eid in ids):
        raise ValueError("引用了没有提供的KG证据")
    if label in {"support", "contradiction"} and not ids:
        raise ValueError("support/contradiction必须有KG证据引用")
    rationale = str(obj.get("rationale") or obj.get("reason") or "").strip()[:500]
    cited = [item for item in task["trusted_kg_evidence"] if item["prompt_evidence_id"] in ids]
    return {"label": label, "rationale": rationale,
            "cited_evidence_ids": ids, "cited_kg_ids": [item["kg_id"] for item in cited],
            "citation_autofilled": False}


def generate(task: dict, url: str, timeout: int, retries: int) -> dict:
    attempts, prediction, method, feedback = [], None, "local_model", None
    if not task["trusted_kg_evidence"]:
        prediction = {"label": "unverifiable", "rationale": "该病例可信KG没有可用事实",
                      "cited_evidence_ids": [], "cited_kg_ids": [], "citation_autofilled": False}
        method = "empty_patient_kg"
    else:
        for number in range(1, retries + 1):
            raw = None
            try:
                raw = local_call(url, model_messages(task, feedback), timeout)
                prediction = validate_prediction(task, parse_json(raw))
                attempts.append({"attempt": number, "raw_response": raw, "error": None})
                break
            except Exception as error:
                feedback = str(error)
                attempts.append({"attempt": number, "raw_response": raw, "error": feedback})
    return {"schema_version": SCHEMA, "claim_id": task["claim_id"],
            "answer_id": task["answer_id"], "question_id": task["question_id"],
            "case_id": task["case_id"], "scenario": task["scenario"],
            "task_sha256": task["task_sha256"],
            "status": "passed" if prediction is not None else "rejected",
            "prediction": prediction, "decision_method": method,
            "trusted_kg_evidence": task["trusted_kg_evidence"],
            "attempts": attempts, "synthetic_only": task["synthetic_only"],
            "requires_manual_review": True}


def valid_saved(task: dict, row: dict) -> bool:
    if (row.get("task_sha256") != task["task_sha256"] or
            row.get("claim_id") != task["claim_id"] or row.get("status") != "passed" or
            row.get("trusted_kg_evidence") != task["trusted_kg_evidence"]):
        return False
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        return False
    try:
        checked = validate_prediction(task, {"label": prediction["label"],
                                             "evidence_ids": prediction["cited_evidence_ids"],
                                             "rationale": prediction.get("rationale", "")})
        return checked == prediction
    except (KeyError, TypeError, ValueError):
        return False


def collect(plan: list[dict], work_dir: Path) -> dict[str, dict]:
    saved = {}
    for task in plan:
        path = work_dir / f"{digest(task['claim_id'])[:24]}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("task_sha256") != task["task_sha256"]:
                raise ValueError(f"{task['claim_id']} 的claim或KG候选已改变；请使用新输出目录")
            saved[task["claim_id"]] = row
    return saved


def summarize(plan: list[dict], saved: dict[str, dict], output_dir: Path) -> dict:
    rows = [saved[task["claim_id"]] for task in plan if task["claim_id"] in saved]
    flat = []
    for task in plan:
        row = saved.get(task["claim_id"], {})
        if not valid_saved(task, row):
            continue
        pred = row["prediction"]
        cited = [item for item in task["trusted_kg_evidence"]
                 if item["prompt_evidence_id"] in pred["cited_evidence_ids"]]
        flat.append({"claim_id": task["claim_id"], "answer_id": task["answer_id"],
                     "question_id": task["question_id"], "case_id": task["case_id"],
                     "scenario": task["scenario"], "claim_text": task["claim_text"],
                     "source_sentence": task["source_sentence"],
                     "label": pred["label"], "rationale": pred["rationale"],
                     "cited_kg_evidence": cited,
                     "decision_method": row["decision_method"],
                     "synthetic_only": task["synthetic_only"]})
    write_jsonl(output_dir / "agent2_results.jsonl", rows)
    write_jsonl(output_dir / "agent2_claim_labels.jsonl", flat)
    counts = Counter(row["label"] for row in flat)
    summary = {"schema_version": SCHEMA, "planned_claims": len(plan),
               "passed_claims": len(flat),
               "rejected_claims": sum(saved.get(task["claim_id"], {}).get("status") == "rejected"
                                      for task in plan),
               "pending_claims": len(plan) - len(rows),
               "prediction_counts": {label: counts[label] for label in LABELS},
               "predictions_without_citation": sum(row["label"] != "unverifiable" and
                   not row["cited_kg_evidence"] for row in flat),
               "empty_patient_kg_decisions": sum(row["decision_method"] == "empty_patient_kg"
                                                 for row in flat),
               "reference_labels_used_during_run": False,
               "kb_documents_read": False, "requires_manual_review": True}
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--claims", type=Path, default=CLAIMS)
    parser.add_argument("--kg-dir", type=Path, default=KG_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-claims", type=int, default=0,
                        help="0处理全部claim；正数仅供院内小批预检")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_claims < 0 or args.max_retries < 1 or not 1 <= args.max_candidates <= 30:
        parser.error("参数范围错误：max-claims>=0，max-retries>=1，max-candidates为1到30")
    plan_path, manifest_path = args.output_dir / "agent2_plan.jsonl", args.output_dir / "plan_manifest.json"
    if args.stage == "prepare":
        if plan_path.exists():
            raise FileExistsError("Agent2计划已冻结；请继续run或使用新输出目录")
        plan, report = make_plan(args.claims, args.kg_dir, args.max_candidates)
        write_jsonl(plan_path, plan)
        write_json(manifest_path, {"schema_version": SCHEMA,
                                   "source_hashes": source_hashes(args.claims, args.kg_dir), **report})
        print(json.dumps(report, ensure_ascii=False))
        return
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("先执行 --stage prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_hashes"] != source_hashes(args.claims, args.kg_dir):
        raise ValueError("Agent1 claim或可信KG已变化；不能混用冻结的Agent2任务")
    plan = read_jsonl(plan_path)
    saved = collect(plan, args.output_dir / "work")
    if args.stage == "run":
        if saved and not args.resume:
            raise ValueError("已有Agent2结果；继续运行请加 --resume")
        for task in plan[:args.max_claims or None]:
            old = saved.get(task["claim_id"])
            if old and valid_saved(task, old):
                continue
            row = generate(task, args.local_url, args.timeout, args.max_retries)
            write_json(args.output_dir / "work" / f"{digest(task['claim_id'])[:24]}.json", row)
            saved[task["claim_id"]] = row
            label = row["prediction"]["label"] if row["prediction"] else "rejected"
            print(f"{task['claim_id']}: {label}", flush=True)
    print(json.dumps(summarize(plan, saved, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
