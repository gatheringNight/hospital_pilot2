"""Agent2A: align routed contradiction claims to actual retrieved KB text.

The local model sees only a claim and numbered spans from the retrieved text.
Document role, Agent2 label, and KG counterevidence are attached afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT2 = ROOT / "private/step6_agent2_natural/agent2_claim_labels.jsonl"
RAG = ROOT / "private/step4_output/rag_results.jsonl"
KB = ROOT / "private/step3_output/kb_documents.jsonl"
KG_DIR = ROOT / "private/step3_output"
OUTPUT = ROOT / "private/step7_agent2a_natural"
LOCAL_URL = "http://127.0.0.1:18080/v1/chat/completions"
SCHEMA = "hospital-local-agent2a-source-attribution-v1"
RELATIONS = ("entails", "contradicts", "partial", "unrelated")
ALIASES = {"蕴含": "entails", "支持": "entails", "一致": "entails",
           "矛盾": "contradicts", "冲突": "contradicts", "不一致": "contradicts",
           "部分": "partial", "部分相关": "partial", "无关": "unrelated"}


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


def source_hashes(agent2_path: Path, rag_path: Path, kb_path: Path, kg_dir: Path) -> dict:
    return {"agent2": file_hash(agent2_path), "rag": file_hash(rag_path),
            "kb_documents": file_hash(kb_path),
            "kg_facts": file_hash(kg_dir / "trusted_kg_facts.jsonl"),
            "kg_changes": file_hash(kg_dir / "trusted_kg_changes.jsonl")}


def unique(rows: list[dict], key: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        item_id = row.get(key)
        if not item_id or item_id in indexed:
            raise ValueError(f"{key}为空或重复：{item_id}")
        indexed[item_id] = row
    return indexed


def text_spans(text: str) -> list[dict]:
    spans = []
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
        raw = match.group()
        trimmed = raw.strip()
        if not trimmed:
            continue
        offset = len(raw) - len(raw.lstrip())
        start = match.start() + offset
        spans.append({"span_id": f"D{len(spans) + 1}", "text": trimmed,
                      "start": start, "end": start + len(trimmed)})
    if not spans:
        raise ValueError("检索文档没有可对齐的文本片段")
    return spans


def numbers(text: str) -> set[Decimal]:
    return {Decimal(value) for value in re.findall(r"\d+(?:\.\d+)?", text)}


def verified_kg_citations(citations: list[dict], case_id: str,
                          facts: dict[str, dict], changes: dict[str, dict]) -> list[dict]:
    verified = []
    for cited in citations:
        kind, kg_id = cited.get("kind"), cited.get("kg_id")
        if kind == "fact":
            fact = facts.get(kg_id)
            statement = (f"{fact['record_date']}记录：{fact['label']}为{fact['object']}。"
                         if fact else None)
            if (not fact or fact["case_id"] != case_id or cited.get("case_id") != case_id or
                    cited.get("object") != fact["object"] or cited.get("label") != fact["label"] or
                    cited.get("record_date") != fact["record_date"] or
                    cited.get("statement") != statement or
                    cited.get("source_facts") != [{"fact_id": kg_id, "source": fact["source"]}]):
                raise ValueError(f"KG事实引用与冻结图谱不一致：{kg_id}")
            verified.append({"kind": "fact", "kg_id": kg_id, "statement": cited["statement"],
                             "source_facts": cited["source_facts"]})
        elif kind == "change":
            change = changes.get(kg_id)
            if not change or change["case_id"] != case_id or cited.get("case_id") != case_id:
                raise ValueError(f"KG变化引用与病例不一致：{kg_id}")
            linked = [facts.get("F" + eid) for eid in change["evidence_ids"]]
            if (any(fact is None or fact["case_id"] != case_id for fact in linked) or
                    cited.get("object") != change["object"] or cited.get("label") != change["label"] or
                    cited.get("source_facts") != [
                        {"fact_id": fact["fact_id"], "source": fact["source"]} for fact in linked]):
                raise ValueError(f"KG变化事实的原始行引用不一致：{kg_id}")
            direction = {"decreased": "下降", "increased": "上升", "unchanged": "不变"}.get(
                change["object"], change["object"])
            statement = (f"{change['label']}由{linked[0]['record_date']}的{change['before']}"
                         f"变为{linked[1]['record_date']}的{change['after']}，方向为{direction}。")
            if cited.get("statement") != statement:
                raise ValueError(f"KG变化事实的陈述与冻结图谱不一致：{kg_id}")
            verified.append({"kind": "change", "kg_id": kg_id, "statement": cited["statement"],
                             "source_facts": cited["source_facts"]})
        else:
            raise ValueError(f"未知KG引用类型：{kind}")
    return verified


def make_plan(agent2_path: Path, rag_path: Path, kb_path: Path,
              kg_dir: Path) -> tuple[list[dict], dict]:
    checks = read_jsonl(agent2_path)
    rag = unique(read_jsonl(rag_path), "answer_id")
    kb = unique(read_jsonl(kb_path), "doc_id")
    facts = unique(read_jsonl(kg_dir / "trusted_kg_facts.jsonl"), "fact_id")
    changes = unique(read_jsonl(kg_dir / "trusted_kg_changes.jsonl"), "change_id")
    routed = [row for row in checks if row.get("label") == "contradiction"]
    plan, seen = [], set()
    for check in routed:
        claim_id, answer_id = check["claim_id"], check["answer_id"]
        if claim_id in seen:
            raise ValueError(f"重复的contradiction claim：{claim_id}")
        seen.add(claim_id)
        answer = rag.get(answer_id)
        if (not answer or answer.get("status") != "passed" or
                any(answer.get(key) != check.get(key) for key in
                    ("question_id", "case_id", "scenario")) or
                check["source_sentence"] not in answer["RAG Answer"]):
            raise ValueError(f"claim无法定位到实际RAG回答：{claim_id}")
        retrieved = answer.get("retrieved") or []
        if len(retrieved) != 1 or retrieved[0].get("doc_id") not in kb:
            raise ValueError(f"{claim_id}没有唯一的实际检索文档")
        context = retrieved[0]
        document = kb[context["doc_id"]]
        if (document["case_id"] != check["case_id"] or document["text"] != context.get("text") or
                text_hash(document["text"]) != context.get("text_sha256") or
                document.get("text_sha256") != context.get("text_sha256") or
                document["doc_role"] != context.get("doc_role")):
            raise ValueError(f"检索文档与冻结KB不一致：{claim_id}")
        citations = check.get("cited_kg_evidence") or []
        if not citations:
            raise ValueError(f"contradiction没有可信KG引用：{claim_id}")
        kg_citations = verified_kg_citations(citations, check["case_id"], facts, changes)
        task = {"schema_version": SCHEMA, "claim_id": claim_id,
                "answer_id": answer_id, "question_id": check["question_id"],
                "case_id": check["case_id"], "scenario": check["scenario"],
                "claim_text": check["claim_text"], "answer_quote": check["source_sentence"],
                "RAG Answer": answer["RAG Answer"],
                "retrieved_source": {"doc_id": document["doc_id"],
                                     "doc_role": document["doc_role"], "text": document["text"],
                                     "text_sha256": context["text_sha256"],
                                     "retrieval_score": context["score"]},
                "source_spans": text_spans(document["text"]),
                "kg_counterevidence": kg_citations,
                "agent2_label": check["label"], "agent2_rationale": check.get("rationale", ""),
                "synthetic_only": check.get("synthetic_only", False)}
        task["task_sha256"] = digest(task)
        plan.append(task)
    report = {"agent2_claims": len(checks), "routed_contradictions": len(routed),
              "planned_source_alignments": len(plan),
              "model_input_fields": ["claim_text", "retrieved_source.text"],
              "hidden_during_alignment": ["doc_role", "Agent2 label", "KG counterevidence"]}
    return plan, report


def model_messages(task: dict, feedback: str | None = None) -> list[dict]:
    system = (
        "你是检索来源文本对齐员。只判断给定检索文本在字面上如何对应claim，"
        "不要判断临床真假，也不要使用外部知识。"
        "文本完整表达claim时为entails，即使该文本本身可能是错的；"
        "文本明确表达相反事实时为contradicts；只覆盖部分为partial；无关为unrelated。"
        "引用一至三个提供的D编号；unrelated时返回空列表。"
        "只返回JSON：{\"relation\":\"entails|contradicts|partial|unrelated\","
        "\"span_ids\":[\"D1\"],\"rationale\":\"简短说明\"}。"
    )
    payload = {"claim": task["claim_text"],
               "retrieved_passage": [{"span_id": item["span_id"], "text": item["text"]}
                                     for item in task["source_spans"]]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if feedback:
        messages.append({"role": "user", "content": "上次格式或引用有误：" + feedback[:240] +
                         "。只使用给定的D编号，重新返回JSON。"})
    return messages


def local_call(url: str, messages: list[dict], timeout: int) -> str:
    target = urllib.parse.urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1"} or not target.port:
        raise ValueError("Agent2A只允许连接本机模型服务")

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
            raise ValueError("模型没有返回JSON对象")
        obj, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(obj, dict):
        raise ValueError("模型输出不是JSON对象")
    return obj


def validate_alignment(task: dict, obj: dict) -> dict:
    relation = str(obj.get("relation") or "").strip().casefold()
    relation = ALIASES.get(relation, relation)
    if relation not in RELATIONS:
        raise ValueError(f"无效来源关系：{relation}")
    ids = obj.get("span_ids", obj.get("span_id", []))
    if isinstance(ids, (str, int)):
        ids = [ids] if str(ids).strip() else []
    if not isinstance(ids, list):
        raise ValueError("span_ids不是列表")
    normalized = []
    for item in ids:
        if isinstance(item, int):
            item = f"D{item}"
        elif isinstance(item, str) and re.fullmatch(r"[dD]?\d+", item.strip()):
            item = "D" + item.strip().lstrip("dD")
        if not isinstance(item, str) or not item.strip():
            raise ValueError("无效来源片段ID")
        normalized.append(item.strip())
    if len(normalized) != len(set(normalized)) or len(normalized) > 3:
        raise ValueError("来源片段ID重复或超过三项")
    spans = {item["span_id"]: item for item in task["source_spans"]}
    if any(item not in spans for item in normalized):
        raise ValueError("引用了不存在的检索文本片段")
    if relation != "unrelated" and not normalized:
        raise ValueError("非unrelated关系须引用文本片段")
    if relation == "unrelated" and normalized:
        raise ValueError("unrelated关系不应引用文本片段")
    selected = [spans[item] for item in normalized]
    if relation == "entails" and not numbers(task["claim_text"]).issubset(
            numbers(" ".join(item["text"] for item in selected))):
        raise ValueError("entails引用片段没有覆盖claim中的数值")
    return {"relation": relation, "span_ids": normalized,
            "source_quotes": [item["text"] for item in selected],
            "source_offsets": [[item["start"], item["end"]] for item in selected],
            "rationale": str(obj.get("rationale") or obj.get("reason") or "").strip()[:500]}


def generate(task: dict, url: str, timeout: int, retries: int) -> dict:
    attempts, alignment, feedback = [], None, None
    for number in range(1, retries + 1):
        raw = None
        try:
            raw = local_call(url, model_messages(task, feedback), timeout)
            alignment = validate_alignment(task, parse_json(raw))
            attempts.append({"attempt": number, "raw_response": raw, "error": None})
            break
        except Exception as error:
            feedback = str(error)
            attempts.append({"attempt": number, "raw_response": raw, "error": feedback})
    role = task["retrieved_source"]["doc_role"]
    chain_complete = bool(alignment and alignment["relation"] == "entails" and
                          alignment["source_quotes"] and task["kg_counterevidence"])
    return {"schema_version": SCHEMA, "claim_id": task["claim_id"],
            "answer_id": task["answer_id"], "question_id": task["question_id"],
            "case_id": task["case_id"], "scenario": task["scenario"],
            "task_sha256": task["task_sha256"],
            "status": "passed" if alignment is not None else "rejected",
            "answer_to_claim": {"RAG Answer": task["RAG Answer"],
                                "claim_text": task["claim_text"],
                                "answer_quote": task["answer_quote"]},
            "claim_to_retrieved_source": {"alignment": alignment,
                                          "doc_id": task["retrieved_source"]["doc_id"],
                                          "doc_role_after_alignment": role,
                                          "text_sha256": task["retrieved_source"]["text_sha256"],
                                          "retrieval_score": task["retrieved_source"]["retrieval_score"]},
            "claim_to_trusted_kg": {"agent2_label": task["agent2_label"],
                                    "agent2_rationale": task["agent2_rationale"],
                                    "kg_counterevidence": task["kg_counterevidence"]},
            "complete_source_chain": chain_complete,
            "poisoned_source_chain": chain_complete and role == "poisoned_doc",
            "source_role_hidden_during_alignment": True,
            "alignment_is_not_causal_proof": True,
            "attempts": attempts, "synthetic_only": task["synthetic_only"]}


def valid_saved(task: dict, row: dict) -> bool:
    if (row.get("task_sha256") != task["task_sha256"] or
            row.get("claim_id") != task["claim_id"] or row.get("status") != "passed"):
        return False
    alignment = row.get("claim_to_retrieved_source", {}).get("alignment")
    if not isinstance(alignment, dict):
        return False
    try:
        checked = validate_alignment(task, {"relation": alignment["relation"],
                                            "span_ids": alignment["span_ids"],
                                            "rationale": alignment.get("rationale", "")})
        source = row["claim_to_retrieved_source"]
        kg = row["claim_to_trusted_kg"]
        expected_complete = bool(checked["relation"] == "entails" and
                                 checked["source_quotes"] and task["kg_counterevidence"])
        return (checked == alignment and
                source["doc_id"] == task["retrieved_source"]["doc_id"] and
                source["doc_role_after_alignment"] == task["retrieved_source"]["doc_role"] and
                source["text_sha256"] == task["retrieved_source"]["text_sha256"] and
                kg["agent2_label"] == task["agent2_label"] and
                kg["kg_counterevidence"] == task["kg_counterevidence"] and
                row["answer_to_claim"]["answer_quote"] == task["answer_quote"] and
                row["complete_source_chain"] == expected_complete and
                row["poisoned_source_chain"] ==
                (expected_complete and task["retrieved_source"]["doc_role"] == "poisoned_doc"))
    except (KeyError, TypeError, ValueError):
        return False


def collect(plan: list[dict], work_dir: Path) -> dict[str, dict]:
    saved = {}
    for task in plan:
        path = work_dir / f"{digest(task['claim_id'])[:24]}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("task_sha256") != task["task_sha256"]:
                raise ValueError(f"{task['claim_id']} 的来源任务已改变；请使用新输出目录")
            saved[task["claim_id"]] = row
    return saved


def summarize(plan: list[dict], saved: dict[str, dict], output_dir: Path) -> dict:
    rows = [saved[task["claim_id"]] for task in plan if task["claim_id"] in saved]
    passed = [saved[task["claim_id"]] for task in plan
              if valid_saved(task, saved.get(task["claim_id"], {}))]
    write_jsonl(output_dir / "source_results.jsonl", rows)
    write_jsonl(output_dir / "source_attributions.jsonl", passed)
    by_answer = defaultdict(list)
    for row in passed:
        by_answer[row["answer_id"]].append(row)
    chains = [{"answer_id": answer_id, "question_id": items[0]["question_id"],
               "claim_ids": [item["claim_id"] for item in items],
               "complete_source_chains": sum(item["complete_source_chain"] for item in items),
               "poisoned_source_chains": sum(item["poisoned_source_chain"] for item in items)}
              for answer_id, items in sorted(by_answer.items())]
    write_jsonl(output_dir / "evidence_chains_by_answer.jsonl", chains)
    relations = Counter(item["claim_to_retrieved_source"]["alignment"]["relation"] for item in passed)
    summary = {"schema_version": SCHEMA, "planned_claims": len(plan),
               "passed_claims": len(passed),
               "rejected_claims": sum(saved.get(task["claim_id"], {}).get("status") == "rejected"
                                      for task in plan),
               "pending_claims": len(plan) - len(rows),
               "source_relation_counts": {relation: relations[relation] for relation in RELATIONS},
               "complete_source_chains": sum(item["complete_source_chain"] for item in passed),
               "poisoned_source_chains": sum(item["poisoned_source_chain"] for item in passed),
               "source_roles_posthoc": dict(Counter(
                   item["claim_to_retrieved_source"]["doc_role_after_alignment"] for item in passed)),
               "model_saw_doc_role_or_kg": False,
               "alignment_is_not_causal_proof": True, "requires_manual_review": True}
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--agent2", type=Path, default=AGENT2)
    parser.add_argument("--rag", type=Path, default=RAG)
    parser.add_argument("--kb-documents", type=Path, default=KB)
    parser.add_argument("--kg-dir", type=Path, default=KG_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=0,
                        help="0处理全部路由claim；正数仅供院内小批预检")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_items < 0 or args.max_retries < 1:
        parser.error("max-items不能为负数，max-retries至少为1")
    plan_path, manifest_path = args.output_dir / "source_plan.jsonl", args.output_dir / "plan_manifest.json"
    if args.stage == "prepare":
        if plan_path.exists():
            raise FileExistsError("Agent2A计划已冻结；请继续run或使用新输出目录")
        plan, report = make_plan(args.agent2, args.rag, args.kb_documents, args.kg_dir)
        write_jsonl(plan_path, plan)
        write_json(manifest_path, {"schema_version": SCHEMA,
                                   "source_hashes": source_hashes(
                                       args.agent2, args.rag, args.kb_documents, args.kg_dir), **report})
        print(json.dumps(report, ensure_ascii=False))
        return
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("先执行 --stage prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_hashes"] != source_hashes(args.agent2, args.rag, args.kb_documents, args.kg_dir):
        raise ValueError("Agent2、RAG、KB或KG已变化；不能混用冻结的Agent2A任务")
    plan = read_jsonl(plan_path)
    saved = collect(plan, args.output_dir / "work")
    if args.stage == "run":
        if saved and not args.resume:
            raise ValueError("已有Agent2A结果；继续运行请加 --resume")
        for task in plan[:args.max_items or None]:
            old = saved.get(task["claim_id"])
            if old and valid_saved(task, old):
                continue
            row = generate(task, args.local_url, args.timeout, args.max_retries)
            write_json(args.output_dir / "work" / f"{digest(task['claim_id'])[:24]}.json", row)
            saved[task["claim_id"]] = row
            relation = row["claim_to_retrieved_source"]["alignment"]
            print(f"{task['claim_id']}: {relation['relation'] if relation else 'rejected'}", flush=True)
    print(json.dumps(summarize(plan, saved, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
