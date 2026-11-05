"""Extract variable-count atomic claims from local RAG answers.

Only Question and RAG Answer are provided to the local model. The source
sentence is attached deterministically from the answer, not copied by the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "private/step4_output/agent1_natural_top1_inputs.jsonl"
OUTPUT = ROOT / "private/step5_agent1_natural"
LOCAL_URL = "http://127.0.0.1:18080/v1/chat/completions"
SCHEMA = "hospital-local-agent1-atomic-v1"
NO_FACT_MARKERS = ("无法确定", "信息不足", "未提供", "没有相关记录", "资料不足", "不能判断")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


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


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def answer_spans(answer: str) -> list[dict]:
    spans = []
    # Decimal points stay inside a span. Each selected substring is literally
    # present in the answer, so later source_sentence cannot be fabricated.
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", answer):
        raw = match.group()
        trimmed = raw.strip()
        if not trimmed:
            continue
        offset = len(raw) - len(raw.lstrip())
        start = match.start() + offset
        spans.append({"span_id": f"S{len(spans) + 1}", "text": trimmed,
                      "start": start, "end": start + len(trimmed)})
    if not spans:
        raise ValueError("回答没有可提取的原文片段")
    return spans


def make_plan(input_path: Path) -> tuple[list[dict], dict]:
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("没有已通过的RAG回答可供Agent1处理")
    plan, seen, scenarios = [], set(), set()
    for row in rows:
        answer_id = row.get("answer_id")
        answer = row.get("RAG Answer")
        if (not answer_id or answer_id in seen or not row.get("question_id") or
                not row.get("case_id") or not row.get("Question") or
                not isinstance(answer, str) or not answer.strip()):
            raise ValueError(f"Agent1输入重复或缺少问题/回答：{answer_id}")
        seen.add(answer_id)
        scenarios.add(row.get("scenario"))
        task = {"schema_version": SCHEMA, "answer_id": answer_id,
                "question_id": row["question_id"], "case_id": row["case_id"],
                "scenario": row["scenario"], "Question": row["Question"],
                "RAG Answer": answer, "spans": answer_spans(answer),
                "synthetic_only": row.get("synthetic_only", False)}
        task["task_sha256"] = digest(task)
        plan.append(task)
    if len(scenarios) != 1 or scenarios.pop() not in {"natural_top1", "clean_only"}:
        raise ValueError("单次Agent1任务必须只含一条回答线")
    return plan, {"answers": len(plan), "source": str(input_path),
                  "model_input_fields": ["Question", "RAG Answer"],
                  "fact_labels_assigned": False, "KG_read": False}


def model_messages(task: dict, feedback: str | None = None) -> list[dict]:
    system = (
        "你是医学回答的原子事实陈述抽取员。只依据给定的RAG回答和问题，"
        "把回答中明确陈述、可独立核查的事实拆成简短完整的claim。"
        "每个claim只讲一个事实；不同时间、不同测量或不同诊断尽量分开。"
        "不判断真假，不使用外部医学知识，不把问题中的未回答内容当事实。"
        "数值、单位、日期和否定词要忠实于回答原文。"
        "每条claim指出其所依据的span_id；无需复制原句。"
        "若回答只有‘无法确定’等非事实性内容，返回空claims。"
        "只返回JSON：{\"claims\":[{\"span_id\":\"S1\",\"claim_text\":\"一条原子事实\"}]}。"
    )
    payload = {"Question": task["Question"],
               "RAG Answer": [{"span_id": span["span_id"], "text": span["text"]}
                              for span in task["spans"]]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if feedback:
        messages.append({"role": "user", "content": "上次格式或原文对应有问题：" + feedback[:240] +
                         "。请重新输出完整JSON，不增加回答原文之外的事实。"})
    return messages


def local_call(url: str, messages: list[dict], timeout: int) -> str:
    target = urllib.parse.urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1"} or not target.port:
        raise ValueError("Agent1只允许连接本机模型服务")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            raise ValueError("本机模型接口重定向已拒绝")

    body = json.dumps({"model": "local-qwen", "messages": messages,
                       "temperature": 0.0, "max_tokens": 1500,
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


def numeric_values(text: str) -> list[Decimal]:
    values = []
    for token in NUMBER.findall(text):
        try:
            values.append(Decimal(token))
        except InvalidOperation:
            pass
    return values


def make_claims(task: dict, parsed: dict) -> tuple[list[dict], list[str]]:
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list) or len(raw_claims) > 24:
        raise ValueError("claims必须是最多24条的列表")
    if (not raw_claims and (not any(marker in task["RAG Answer"] for marker in NO_FACT_MARKERS)
                            or numeric_values(task["RAG Answer"]))):
        raise ValueError("包含事实内容的回答不能返回空claims")
    spans = {span["span_id"]: span for span in task["spans"]}
    claims, seen, warnings, covered = [], set(), [], set()
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise ValueError("claim不是JSON对象")
        span_id = raw.get("span_id")
        if isinstance(span_id, int):
            span_id = f"S{span_id}"
        elif isinstance(span_id, str) and re.fullmatch(r"[sS]?\d+", span_id.strip()):
            span_id = "S" + span_id.strip().lstrip("sS")
        if span_id not in spans:
            raise ValueError(f"claim引用不存在的回答片段：{span_id}")
        text = raw.get("claim_text", raw.get("claim"))
        if not isinstance(text, str) or not 4 <= len(text.strip()) <= 240:
            raise ValueError("claim_text为空或过长")
        text = text.strip()
        key = re.sub(r"\s+", "", text).rstrip("。；，").casefold()
        if key in seen:
            continue
        seen.add(key)
        span = spans[span_id]
        if any(number not in numeric_values(span["text"]) for number in numeric_values(text)):
            raise ValueError(f"{span_id}的claim含回答原句中没有的数值")
        quote = raw.get("source_quote")
        if not isinstance(quote, str) or not quote.strip() or quote.strip() not in span["text"]:
            quote = span["text"]
        else:
            quote = quote.strip()
        covered.add(span_id)
        claims.append({"schema_version": SCHEMA,
                       "claim_id": f"{task['answer_id']}_c{len(claims) + 1}",
                       "answer_id": task["answer_id"], "question_id": task["question_id"],
                       "case_id": task["case_id"], "scenario": task["scenario"],
                       "Question": task["Question"],
                       "claim_text": text, "source_sentence": span["text"],
                       "source_quote": quote, "span_id": span_id,
                       "source_start": span["start"], "source_end": span["end"],
                       "synthetic_only": task["synthetic_only"]})
    uncovered = [span_id for span_id, span in spans.items()
                 if span_id not in covered and len(span["text"]) >= 12 and
                 not any(marker in span["text"] for marker in NO_FACT_MARKERS)]
    if uncovered:
        warnings.append("可能未覆盖的回答片段：" + ",".join(uncovered))
    return claims, warnings


def generate(task: dict, url: str, timeout: int, retries: int) -> dict:
    attempts, claims, warnings, feedback = [], None, [], None
    for number in range(1, retries + 1):
        raw = None
        try:
            raw = local_call(url, model_messages(task, feedback), timeout)
            claims, warnings = make_claims(task, parse_json(raw))
            attempts.append({"attempt": number, "raw_response": raw, "error": None})
            break
        except Exception as error:
            feedback = str(error)
            attempts.append({"attempt": number, "raw_response": raw, "error": feedback})
    return {"schema_version": SCHEMA, "answer_id": task["answer_id"],
            "task_sha256": task["task_sha256"], "question_id": task["question_id"],
            "case_id": task["case_id"], "scenario": task["scenario"],
            "status": "passed" if claims is not None else "rejected",
            "claims": claims or [], "warnings": warnings, "attempts": attempts,
            "synthetic_only": task["synthetic_only"], "requires_manual_review": True}


def valid_saved(task: dict, row: dict) -> bool:
    if row.get("task_sha256") != task["task_sha256"] or row.get("status") != "passed":
        return False
    try:
        rebuilt, _ = make_claims(task, {"claims": row["claims"]})
    except (KeyError, TypeError, ValueError):
        return False
    return rebuilt == row["claims"]


def collect(plan: list[dict], work_dir: Path) -> dict[str, dict]:
    saved = {}
    for task in plan:
        path = work_dir / f"{task['answer_id']}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("task_sha256") != task["task_sha256"]:
                raise ValueError(f"{task['answer_id']} 的原回答或任务已改变；请使用新输出目录")
            saved[task["answer_id"]] = row
    return saved


def summarize(plan: list[dict], saved: dict[str, dict], output_dir: Path) -> dict:
    rows = [saved[task["answer_id"]] for task in plan if task["answer_id"] in saved]
    flat = [claim for task in plan if valid_saved(task, saved.get(task["answer_id"], {}))
            for claim in saved[task["answer_id"]]["claims"]]
    write_jsonl(output_dir / "agent1_results.jsonl", rows)
    write_jsonl(output_dir / "agent1_atomic_claims.jsonl", flat)
    passed = sum(valid_saved(task, saved.get(task["answer_id"], {})) for task in plan)
    summary = {"schema_version": SCHEMA, "planned_answers": len(plan),
               "passed_answers": passed, "rejected_answers": sum(
                   saved.get(task["answer_id"], {}).get("status") == "rejected" for task in plan),
               "pending_answers": len(plan) - len(rows), "atomic_claims": len(flat),
               "answers_with_no_factual_claims": sum(
                   valid_saved(task, saved.get(task["answer_id"], {})) and
                   not saved[task["answer_id"]]["claims"] for task in plan),
               "review_warning_counts": dict(Counter(
                   warning.split("：", 1)[0] for row in rows for warning in row.get("warnings", []))),
               "model_input_fields": ["Question", "RAG Answer"],
               "fact_labels_assigned": False, "KG_read": False,
               "requires_manual_review": True}
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-answers", type=int, default=0,
                        help="0处理全部回答；正数仅供院内小批预检")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_answers < 0 or args.max_retries < 1:
        parser.error("max-answers不能为负数，max-retries至少为1")
    plan_path, manifest_path = args.output_dir / "agent1_plan.jsonl", args.output_dir / "plan_manifest.json"
    if args.stage == "prepare":
        if plan_path.exists():
            raise FileExistsError("Agent1计划已冻结；请继续run或使用新输出目录")
        plan, report = make_plan(args.input)
        write_jsonl(plan_path, plan)
        write_json(manifest_path, {"schema_version": SCHEMA,
                                   "input_sha256": file_digest(args.input), **report})
        print(json.dumps(report, ensure_ascii=False))
        return
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("先执行 --stage prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != file_digest(args.input):
        raise ValueError("RAG回答文件已变化；不能混用已冻结的Agent1任务")
    plan = read_jsonl(plan_path)
    saved = collect(plan, args.output_dir / "work")
    if args.stage == "run":
        if saved and not args.resume:
            raise ValueError("已有Agent1结果；继续运行请加 --resume")
        for task in plan[:args.max_answers or None]:
            old = saved.get(task["answer_id"])
            if old and valid_saved(task, old):
                continue
            row = generate(task, args.local_url, args.timeout, args.max_retries)
            write_json(args.output_dir / "work" / f"{task['answer_id']}.json", row)
            saved[task["answer_id"]] = row
            print(f"{task['question_id']}: {row['status']} claims={len(row['claims'])}", flush=True)
    print(json.dumps(summarize(plan, saved, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
