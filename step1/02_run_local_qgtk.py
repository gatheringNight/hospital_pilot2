"""Generate Q/GT/K from prepared evidence-card tasks with a local model only.

All patient/encounter linkage and evidence selection are finished by 01. This
stage only verbalizes each frozen task, validates source values, and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "private/step1_output"
DEFAULT_URL = "http://127.0.0.1:18080/v1/chat/completions"


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"先运行01准备任务：{path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("％", "%").casefold()


def contains_value(text: str, value: str) -> bool:
    return re.search(r"(?<!\d)" + re.escape(compact(value)) + r"(?!\d)", compact(text)) is not None


def model_cards(cards: list[dict]) -> list[dict]:
    return [{key: card[key] for key in ("evidence_id", "label", "fact", "date", "visit_label")}
            for card in cards]


def date_list(cards: list[dict]) -> list[str]:
    return list(dict.fromkeys(card["date"] for card in cards))


def messages_for(task: dict, stage: str, cards: list[dict] | None = None,
                 knowledge: list[str] | None = None) -> list[dict]:
    common = ("你是院内本地记录整理助手，只依据给出的证据卡写文字。"
              "证据卡来自同一病例，患者身份和原始流水号未提供。"
              "每张卡的名称、数值、单位和记录日期必须保留，不能补充诊断、风险、疗效或治疗建议。"
              "不要重复指令，只返回要求的JSON对象。")
    if stage == "knowledge":
        system = (common + "把这些卡片整合成一段连贯的Knowledge文字，明确各事实属于哪次记录。"
                  "使用自然中文，可以调整语序，但每张卡的原值和主题必须能看出。"
                  "只返回{\"paragraph\":\"一段文字\"}。")
        payload = {"evidence_cards": model_cards(cards or [])}
    elif stage == "answer":
        system = (common + "根据下列事实直接回答问题意图，写完整、自然的Ground Truth，"
                  "保留每条原始值及单位。只描述记录所载事实；两次就诊的值不可混写。"
                  "只返回{\"Ground Truth\":\"完整回答\"}。")
        payload = {"question_intent": task["question_intent"],
                   "evidence_cards": model_cards(task["evidence_cards"]),
                   "accepted_Knowledge": knowledge}
    elif stage == "question":
        system = (common + "根据question_intent写一句独立可读的真实用户问题。"
                  "只问本题已给证据能够回答的内容，点明主题和记录日期；"
                  "不要把任何检查结果或答案数值写进问题，不问原因、诊断或治疗建议。"
                  "只返回{\"Question\":\"完整问题？\"}。")
        payload = {"question_intent": task["question_intent"],
                   "topic": task["topic"], "focus": task["question_focus"],
                   "record_dates": date_list(task["evidence_cards"])}
    else:
        raise ValueError(stage)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def parse_json_object(raw: str) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            raise ValueError("模型回复中没有JSON对象")
        obj, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(obj, dict):
        raise ValueError("模型回复不是JSON对象")
    return obj


def local_call(url: str, messages: list[dict], timeout: int) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port:
        raise ValueError("只允许连接本机HTTP推理服务")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            raise ValueError("本机模型服务返回重定向，已拒绝")

    request_body = json.dumps({"model": "local-qwen", "messages": messages,
                               "temperature": 0.15, "max_tokens": 1700,
                               "response_format": {"type": "json_object"}},
                              ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=request_body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def check_piece(stage: str, text: object, cards: list[dict], task: dict) -> tuple[list[str], list[str]]:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return ["文字缺失或过短"], []
    issues = [f"缺少来源原值：{card['evidence_id']}" for card in cards
              if not contains_value(text, card["required_value"])]
    review = [f"主题词需人工复核：{card['evidence_id']}" for card in cards
              if len(card["label"]) >= 3 and compact(card["label"]) not in compact(text)]
    if stage == "question":
        issues = []
        if not text.endswith(("?", "？")):
            issues.append("Question不是问句")
        if not any(word and word in text for word in task["question_keywords"]):
            issues.append("Question没有本题主题")
        for day in date_list(cards):
            if day not in text:
                issues.append(f"Question缺少记录日期：{day}")
        without_dates = text
        for day in date_list(cards):
            without_dates = without_dates.replace(day, "")
        if re.search(r"\d", without_dates):
            issues.append("Question含日期以外的数值")
        if re.search(r"原因|治疗建议|如何治疗|诊断依据", text):
            issues.append("Question超出当前记录")
        review = []
    else:
        if re.search(r"建议|疗效|风险.{0,8}(降低|升高)|可以诊断|正常范围", text):
            review.append("文字中可能出现记录外临床判断")
    if task["case_id"] in text:
        issues.append("输出含内部病例别名")
    return issues, review


def run_component(task: dict, stage: str, cards: list[dict], url: str, timeout: int,
                  knowledge: list[str] | None = None) -> tuple[str | None, list[dict], list[str]]:
    base = messages_for(task, stage, cards, knowledge)
    field = {"knowledge": "paragraph", "answer": "Ground Truth", "question": "Question"}[stage]
    attempts = []
    for number in range(1, 4):
        try:
            raw = local_call(url, base, timeout)
            value = parse_json_object(raw).get(field)
            if isinstance(value, str):
                value = re.sub(r"^\s*(?:第[一二三]段|段落[一二三])\s*[:：]?\s*", "", value.strip())
            repairs = []
            if stage == "knowledge" and isinstance(value, str):
                for day in date_list(cards):
                    if day not in value:
                        value = f"{day}记录：{value}"
                        repairs.append(f"补入证据卡日期：{day}")
            if stage == "question" and isinstance(value, str):
                if value and not value.endswith(("?", "？")):
                    value += "？"
                    repairs.append("补入问号")
                if not re.search(r"\d", value):
                    missing = [day for day in date_list(cards) if day not in value]
                    if missing:
                        value = f"对照{'与'.join(missing)}记录，{value}"
                        repairs.append("补入证据卡日期")
            issues, review = check_piece(stage, value, cards, task)
            attempts.append({"stage": stage, "attempt": number, "raw_response": raw,
                             "source_only_repairs": repairs, "issues": issues, "review_flags": review})
            if not issues:
                return value, attempts, review
            base = messages_for(task, stage, cards, knowledge) + [
                {"role": "user", "content": f"上次{field}未通过：{'；'.join(issues)}。只重新输出该字段的JSON对象。"}]
        except Exception as error:
            attempts.append({"stage": stage, "attempt": number, "error": str(error)})
    return None, attempts, []


def inspect_result(task: dict, result: dict) -> tuple[list[str], list[str]]:
    issues, review = [], []
    if (result.get("source_signature") != task["source_signature"] or
            result.get("question_id") != task["question_id"] or
            result.get("evidence_card_ids") != [card["evidence_id"] for card in task["evidence_cards"]]):
        issues.append("任务来源或证据卡集合已变化")
    by_id = {card["evidence_id"]: card for card in task["evidence_cards"]}
    knowledge = result.get("Knowledge")
    if not isinstance(knowledge, list) or len(knowledge) != len(task["knowledge_groups"]):
        issues.append("Knowledge段落数与01准备任务不一致")
    else:
        for paragraph, ids in zip(knowledge, task["knowledge_groups"]):
            a, b = check_piece("knowledge", paragraph, [by_id[eid] for eid in ids], task)
            issues += a
            review += b
    a, b = check_piece("answer", result.get("Ground Truth"), task["evidence_cards"], task)
    issues += a
    review += b
    a, b = check_piece("question", result.get("Question"), task["evidence_cards"], task)
    issues += a
    review += b
    return sorted(set(issues)), sorted(set(review))


def generate(task: dict, args) -> dict:
    cards_by_id = {card["evidence_id"]: card for card in task["evidence_cards"]}
    attempts, review, paragraphs = [], [], []
    for ids in task["knowledge_groups"]:
        cards = [cards_by_id[eid] for eid in ids]
        paragraph, history, flags = run_component(task, "knowledge", cards, args.local_url, args.timeout)
        attempts += history
        review += flags
        if paragraph is None:
            break
        paragraphs.append(paragraph)
    ground_truth = question = None
    if len(paragraphs) == len(task["knowledge_groups"]):
        ground_truth, history, flags = run_component(task, "answer", task["evidence_cards"],
                                                     args.local_url, args.timeout, paragraphs)
        attempts += history
        review += flags
    if ground_truth is not None:
        question, history, flags = run_component(task, "question", task["evidence_cards"],
                                                 args.local_url, args.timeout)
        attempts += history
        review += flags
    result = {"schema_version": "hospital-qgtk-result-v2", "question_id": task["question_id"],
              "case_id": task["case_id"], "task_type": task["task_type"],
              "synthetic_only": task["synthetic_only"], "source_signature": task["source_signature"],
              "Question": question, "Ground Truth": ground_truth,
              "Knowledge": paragraphs if len(paragraphs) == len(task["knowledge_groups"]) else None,
              "ground_truth_claims": task["required_claims"],
              "evidence_cards": task["evidence_cards"],
              "evidence_card_ids": [card["evidence_id"] for card in task["evidence_cards"]],
              "attempts": attempts, "requires_manual_review": True}
    issues, flags = inspect_result(task, result)
    result["status"] = "passed" if not issues else "rejected"
    result["validation_issues"] = issues
    result["review_flags"] = sorted(set(review + flags))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "audit"), required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "qgtk_tasks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--max-tasks", type=int, default=0, help="0为全部任务；正数仅供小批预检")
    parser.add_argument("--local-url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tasks < 0:
        raise ValueError("--max-tasks不能为负数")
    tasks = list(read_jsonl(args.input))
    if not tasks or len({task["question_id"] for task in tasks}) != len(tasks):
        raise ValueError("任务为空或问题ID重复；先检查01输出")
    signatures = {task["source_signature"] for task in tasks}
    if len(signatures) != 1:
        raise ValueError("任务包含多个来源批次")
    signature = next(iter(signatures))
    work_dir = args.output_dir / "qgtk_work" / signature[:12]
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        selected = tasks if args.max_tasks == 0 else tasks[:args.max_tasks]
        for index, task in enumerate(selected, 1):
            result_path = work_dir / f"{task['question_id']}.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("status") == "passed" and not inspect_result(task, previous)[0]:
                    continue
                if not args.resume:
                    raise ValueError(f"{task['question_id']}已有记录；继续须加--resume")
            result = generate(task, args)
            save_json(result_path, result)
            print(f"{index}/{len(selected)} {task['question_id']}: {result['status']}", flush=True)
    results_path = args.output_dir / "qgtk_results.jsonl"
    temporary = results_path.with_suffix(".jsonl.tmp")
    counts = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for task in tasks:
            result_path = work_dir / f"{task['question_id']}.json"
            if not result_path.exists():
                counts["pending"] += 1
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            issues, flags = inspect_result(task, result)
            if issues:
                result["status"] = "rejected"
                result["validation_issues"] = issues
            result["review_flags"] = sorted(set(result.get("review_flags", []) + flags))
            counts[result["status"]] += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(results_path)
    summary = {"schema_version": "hospital-qgtk-result-v2", "planned_tasks": len(tasks),
               "passed": counts["passed"], "rejected": counts["rejected"], "pending": counts["pending"],
               "source_signature": signature, "model_called_in_this_command": args.stage == "run",
               "requires_manual_review": True, "output": str(results_path)}
    save_json(args.output_dir / "qgtk_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
