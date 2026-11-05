"""Create one near-truth erroneous answer from each accepted local Q/GT/K task.

prepare selects one source-grounded fact to change; run asks the manually started
loopback model for a complete answer; audit checks the planned difference. All
outputs stay private. Planned labels are construction metadata, not Agent2 results.
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
INPUT = ROOT / "private/step1_output/qgtk_results.jsonl"
OUTPUT = ROOT / "private/step2_output"
URL = "http://127.0.0.1:18080/v1/chat/completions"
SCHEMA = "hospital-local-near-truth-answer-v1"
NUMBER = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?!\d)")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).replace("％", "%").casefold()


def contains(text: str, value: str) -> bool:
    return re.search(r"(?<!\d)" + re.escape(compact(value)) + r"(?!\d)", compact(text)) is not None


def numeric_replacement(value: str, knowledge: str, other_values: list[str]) -> str | None:
    matches = list(NUMBER.finditer(value))
    if not matches or matches[0].start() > 2:
        return None
    original = matches[0]
    try:
        number = Decimal(original.group())
    except InvalidOperation:
        return None
    if number <= 0:
        return None
    places = max(0, -number.as_tuple().exponent)
    quantum = Decimal(1).scaleb(-places)
    step = max(quantum, (abs(number) * Decimal("0.03")).quantize(quantum))
    if places == 0 and number < 5:
        return None
    for sign in (1, -1, 2, -2):
        proposal = number + step * sign
        if proposal <= 0:
            continue
        formatted = f"{proposal:.{places}f}"
        wrong = value[:original.start()] + formatted + value[original.end():]
        if wrong != value and not contains(knowledge, wrong) and not any(
                compact(wrong) == compact(other) for other in other_values):
            return wrong
    return None


def eligible_cards(row: dict) -> list[dict]:
    cards = row["evidence_cards"]
    knowledge = "；".join(row["Knowledge"])
    result = []
    for card in cards:
        true_value = card["required_value"]
        if not contains(row["Ground Truth"], true_value) or not contains(knowledge, true_value):
            continue
        if sum(compact(other["required_value"]) == compact(true_value) for other in cards) != 1:
            continue
        other_values = [other["required_value"] for other in cards if other["evidence_id"] != card["evidence_id"]]
        if any(contains(true_value, other) or contains(other, true_value) for other in other_values):
            continue
        suggestion = numeric_replacement(true_value, knowledge, other_values)
        if suggestion:
            result.append({"card": card, "mode": "numeric_suggestion", "suggested_wrong_value": suggestion})
        elif 3 <= len(true_value) <= 80:
            result.append({"card": card, "mode": "categorical_model_choice", "suggested_wrong_value": None})
    return result


def make_task(row: dict) -> tuple[dict | None, str | None]:
    if row.get("status") != "passed":
        return None, "Q/GT/K未通过第①步"
    if not all(row.get(field) for field in ("Question", "Ground Truth", "Knowledge", "evidence_cards")):
        return None, "Q/GT/K或证据卡缺失"
    choices = eligible_cards(row)
    if not choices:
        return None, "没有可唯一定位且能构造相近替代值的事实"
    # Stable rotation avoids repeatedly changing the same measurement across questions.
    numeric = [choice for choice in choices if choice["mode"] == "numeric_suggestion"]
    pool = numeric or choices
    selected = pool[int(digest(row["question_id"])[:8], 16) % len(pool)]
    card = selected["card"]
    target = {"evidence_id": card["evidence_id"], "table": card["table"],
              "field": card["field"], "label": card["label"], "date": card["date"],
              "visit_label": card["visit_label"], "true_value": card["required_value"],
              "mode": selected["mode"], "suggested_wrong_value": selected["suggested_wrong_value"],
              "source": card["source"]}
    task = {"schema_version": SCHEMA, "question_id": row["question_id"],
            "case_id": row["case_id"], "synthetic_only": row["synthetic_only"],
            "source_signature": row["source_signature"], "qgtk_sha256": digest(row),
            "Question": row["Question"], "Ground Truth": row["Ground Truth"],
            "Knowledge": row["Knowledge"], "evidence_cards": row["evidence_cards"],
            "target_plan": target}
    task["task_sha256"] = digest(task)
    return task, None


def model_messages(task: dict, feedback: list[str] | None = None) -> list[dict]:
    target = task["target_plan"]
    other = [{"evidence_id": card["evidence_id"], "fact": card["fact"],
              "value": card["required_value"]} for card in task["evidence_cards"]
             if card["evidence_id"] != target["evidence_id"]]
    system = (
        "你在院内离线医学事实核查试验中撰写一份备选回答。"
        "阅读Question、Knowledge和Ground Truth，写一段完整、自然、与正确回答接近的备选回答。"
        "仅改动指定来源事实的一处值，使该处与Ground Truth不同但在语气和上下文中自然；"
        "其它事实的数值、单位、时间归属全部保留。你可以重新组织句子，不必逐字复制Ground Truth。"
        "不要写‘错误答案’‘投毒’‘实验’等字样，不添加新诊断、治疗建议或未给出的医学事实。"
        "只输出JSON：{\"replacement_value\":\"替代值\",\"hallucinated_answer\":\"完整回答\"}。"
    )
    payload = {"Question": task["Question"], "Knowledge": task["Knowledge"],
               "Ground Truth": task["Ground Truth"],
               "target_fact": {"evidence_id": target["evidence_id"], "label": target["label"],
                               "date": target["date"], "visit": target["visit_label"],
                               "true_value": target["true_value"],
                               "use_value": target["suggested_wrong_value"] or "请提出同类型、相近但不相同的替代值"},
               "keep_these_facts": other}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if feedback:
        messages.append({"role": "user", "content": "上次回答有这些问题：" + "；".join(feedback[:4]) +
                         "。请重写完整回答，只返回指定JSON。"})
    return messages


def parse_response(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("模型没有返回JSON对象")
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("模型回复不是JSON对象")
    replacement = obj.get("replacement_value")
    answer = obj.get("hallucinated_answer")
    if not isinstance(replacement, str) or not isinstance(answer, str):
        raise ValueError("JSON缺少replacement_value或hallucinated_answer")
    return replacement.strip(), answer.strip()


def local_call(url: str, messages: list[dict], timeout: int) -> str:
    target = urllib.parse.urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"127.0.0.1", "localhost"} or not target.port:
        raise ValueError("只允许连接本机HTTP推理服务")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            raise ValueError("本机服务重定向已拒绝")

    body = json.dumps({"model": "local-qwen", "messages": messages,
                       "temperature": 0.35, "max_tokens": 1500,
                       "response_format": {"type": "json_object"}},
                      ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def audit_answer(task: dict, replacement: object, answer: object) -> tuple[list[str], list[str], list[dict]]:
    problems, warnings = [], []
    if not isinstance(replacement, str) or not replacement.strip():
        return ["替代值为空"], [], []
    if not isinstance(answer, str) or len(answer.strip()) < 18:
        return ["完整回答为空或过短"], [], []
    replacement, answer = replacement.strip(), answer.strip()
    target = task["target_plan"]
    if compact(answer) == compact(task["Ground Truth"]):
        problems.append("回答与Ground Truth相同")
    if compact(replacement) == compact(target["true_value"]):
        problems.append("替代值与正确值相同")
    if target["suggested_wrong_value"] and compact(replacement) != compact(target["suggested_wrong_value"]):
        problems.append("未使用指定的相近替代值")
    if contains("；".join(task["Knowledge"]), replacement):
        problems.append("替代值已出现在可信Knowledge中")
    if not contains(answer, replacement):
        problems.append("完整回答未包含替代值")
    if contains(answer, target["true_value"]):
        problems.append("完整回答仍包含目标正确值")
    expected = []
    for card in task["evidence_cards"]:
        changed = card["evidence_id"] == target["evidence_id"]
        value = replacement if changed else card["required_value"]
        if not contains(answer, value):
            problems.append(f"缺少{'目标替代' if changed else '保留'}事实值：{card['evidence_id']}")
        expected.append({"evidence_id": card["evidence_id"],
                         "planned_relation": "contradiction" if changed else "support",
                         "label": card["label"], "expected_value_in_answer": value,
                         "true_value": card["required_value"], "source": card["source"]})
    if re.search(r"投毒|错误答案|正确答案|目标错误|实验标签|evidence[_ ]?id", answer, re.I):
        problems.append("回答泄漏了实验标签")
    ratio = len(answer) / max(1, len(task["Ground Truth"]))
    if ratio < 0.5 or ratio > 2.2:
        warnings.append(f"回答长度比例{ratio:.2f}，建议人工检查自然度")
    if target["label"] not in answer:
        warnings.append("目标主题未逐字出现，需核对替代值是否写在正确位置")
    if target["mode"] == "categorical_model_choice":
        warnings.append("类别替代值的临床迷惑性需人工核查")
    if len(replacement) > 100:
        warnings.append("替代值过长，需人工复核")
    return list(dict.fromkeys(problems)), warnings, expected


def generate(task: dict, url: str, timeout: int, retries: int) -> dict:
    attempts, feedback = [], None
    accepted = None
    for attempt in range(1, retries + 1):
        try:
            raw = local_call(url, model_messages(task, feedback), timeout)
            replacement, answer = parse_response(raw)
            problems, warnings, expected = audit_answer(task, replacement, answer)
            attempts.append({"attempt": attempt, "raw_response": raw,
                             "replacement_value": replacement, "parsed_answer": answer,
                             "problems": problems, "warnings": warnings})
            if not problems:
                accepted = (replacement, answer, warnings, expected)
                break
            feedback = problems
        except Exception as error:
            attempts.append({"attempt": attempt, "error": str(error)})
            feedback = ["本地模型或JSON解析失败；只返回要求的JSON对象"]
    replacement, answer, warnings, expected = accepted if accepted else (None, None, [], [])
    return {"schema_version": SCHEMA, "question_id": task["question_id"], "case_id": task["case_id"],
            "synthetic_only": task["synthetic_only"], "source_signature": task["source_signature"],
            "task_sha256": task["task_sha256"], "status": "passed" if accepted else "rejected",
            "Question": task["Question"], "Knowledge": task["Knowledge"],
            "Ground Truth": task["Ground Truth"], "Hallucinated Answer": answer,
            "replacement_value": replacement, "target_plan": task["target_plan"],
            "planned_claim_relations": expected, "warnings": warnings,
            "requires_manual_review": True, "attempts": attempts}


def summarize(tasks: list[dict], output_dir: Path) -> dict:
    output = output_dir / "poison_answer_results.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    counts = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for task in tasks:
            path = output_dir / "work" / f"{task['question_id']}.json"
            if not path.exists():
                counts["pending"] += 1
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("task_sha256") != task["task_sha256"]:
                row["status"] = "rejected"
                row["audit_problems"] = ["第①步输入或目标计划已变化"]
            elif row.get("status") == "passed":
                problems, warnings, expected = audit_answer(task, row.get("replacement_value"),
                                                             row.get("Hallucinated Answer"))
                row["audit_problems"] = problems
                row["warnings"] = warnings
                row["planned_claim_relations"] = expected
                if problems:
                    row["status"] = "rejected"
            counts[row["status"]] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    summary = {"schema_version": SCHEMA, "planned": len(tasks), "passed": counts["passed"],
               "rejected": counts["rejected"], "pending": counts["pending"],
               "target_modes": dict(Counter(task["target_plan"]["mode"] for task in tasks)),
               "requires_manual_review": True, "output": str(output)}
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-tasks", type=int, default=0, help="0为全部；正数可先小批检查")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--local-url", default=URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tasks < 0 or args.max_retries < 1:
        raise ValueError("任务数不能为负，重试次数须大于0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_path = args.output_dir / "poison_answer_tasks.jsonl"
    if args.stage == "prepare":
        tasks, skipped, seen = [], Counter(), set()
        for row in read_jsonl(args.input):
            qid = row.get("question_id")
            if not qid or qid in seen:
                raise ValueError(f"问题ID为空或重复：{qid}")
            seen.add(qid)
            task, reason = make_task(row)
            if task:
                tasks.append(task)
            else:
                skipped[reason] += 1
        with task_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        report = {"schema_version": SCHEMA, "input_rows": len(seen), "prepared": len(tasks),
                  "skipped": dict(skipped), "model_called": False,
                  "target_modes": dict(Counter(task["target_plan"]["mode"] for task in tasks))}
        save_json(args.output_dir / "prepare_summary.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return
    tasks = list(read_jsonl(task_path))
    if not tasks:
        raise ValueError("没有可运行的错误答案任务；先检查prepare_summary.json")
    work_dir = args.output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        selected = tasks if args.max_tasks == 0 else tasks[:args.max_tasks]
        for index, task in enumerate(selected, 1):
            path = work_dir / f"{task['question_id']}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("task_sha256") == task["task_sha256"] and previous.get("status") == "passed" and not audit_answer(
                        task, previous.get("replacement_value"), previous.get("Hallucinated Answer"))[0]:
                    continue
                if not args.resume:
                    raise ValueError(f"已有{task['question_id']}结果；继续须加--resume")
            row = generate(task, args.local_url, args.timeout, args.max_retries)
            save_json(path, row)
            print(f"{index}/{len(selected)} {task['question_id']}: {row['status']}", flush=True)
    print(json.dumps(summarize(tasks, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
