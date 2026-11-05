"""Build the local mixed clean/poison KB and trusted source-grounded KG.

Offline only. The KG builder accepts Q/GT/K and original-row provenance, never
the erroneous answer, target plan, document role, or later RAG output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import joblib
import networkx as nx
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
QGTK = ROOT / "private/step1_output/qgtk_results.jsonl"
ANSWERS = ROOT / "private/step2_output/poison_answer_results.jsonl"
SOURCE_DIR = ROOT / "private/step1_output"
OUTPUT = ROOT / "private/step3_output"
SCHEMA = "hospital-local-kb-kg-v2"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).replace("％", "%").casefold()


def contains(text: str, value: str) -> bool:
    return re.search(r"(?<!\d)" + re.escape(compact(value)) + r"(?!\d)", compact(text)) is not None


def unique(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        qid = row.get("question_id")
        if not qid or qid in result:
            raise ValueError(f"{label}问题ID为空或重复：{qid}")
        result[qid] = row
    return result


def paired_inputs(qgtk_rows: list[dict], answer_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    qgtk, answers = unique(qgtk_rows, "Q/GT/K"), unique(answer_rows, "错误答案")
    pairs, excluded = [], Counter()
    for qid, answer in answers.items():
        trusted = qgtk.get(qid)
        if answer.get("status") != "passed" or not trusted or trusted.get("status") != "passed":
            excluded["missing_or_unpassed_pair"] += 1
            continue
        if (answer.get("qgtk_sha256") != digest(trusted) or
                any(answer.get(key) != trusted.get(key) for key in (
                    "case_id", "source_signature", "synthetic_only", "Question", "Ground Truth", "Knowledge"))):
            excluded["frozen_qgtk_mismatch"] += 1
            continue
        cards = trusted.get("evidence_cards")
        if (not isinstance(cards, list) or not cards or
                any(not isinstance(card, dict) or not card.get("evidence_id") for card in cards) or
                len({card["evidence_id"] for card in cards}) != len(cards)):
            excluded["invalid_evidence_cards"] += 1
            continue
        target = answer.get("target_plan", {})
        if not isinstance(target, dict):
            excluded["unbound_target"] += 1
            continue
        by_id = {card["evidence_id"]: card for card in cards}
        card = by_id.get(target.get("evidence_id"))
        if not card or target.get("true_value") != card["required_value"]:
            excluded["unbound_target"] += 1
            continue
        wrong = answer.get("replacement_value")
        poison = answer.get("Hallucinated Answer")
        if (not isinstance(wrong, str) or not isinstance(poison, str)
                or not contains(poison, wrong) or contains(poison, card["required_value"])):
            excluded["target_difference_not_verified"] += 1
            continue
        knowledge_text = "；".join(trusted["Knowledge"])
        grounded = True
        for item in cards:
            value = item["required_value"]
            if not contains(trusted["Ground Truth"], value) or not contains(knowledge_text, value):
                grounded = False
                break
            if item["evidence_id"] != card["evidence_id"] and not contains(poison, value):
                grounded = False
                break
        if not grounded:
            excluded["incomplete_fact_coverage"] += 1
            continue
        pairs.append(trusted)
    if not pairs:
        raise ValueError("没有完整通过的Q/GT/K与错误答案配对")
    report = {"qgtk_rows": len(qgtk), "answer_rows": len(answers),
              "paired": len(pairs), "excluded_answers": dict(excluded),
              "qgtk_without_passed_answer": len(qgtk) - len(pairs)}
    return sorted(pairs, key=lambda row: row["question_id"]), answers, report


def verify_original_rows(qgtk_rows: list[dict], source_dir: Path) -> dict:
    signatures = {row["source_signature"] for row in qgtk_rows}
    if len(signatures) != 1:
        raise ValueError("配对样本来自多个原始数据批次")
    signature = next(iter(signatures))
    source_db = source_dir / f"source_index_{signature[:12]}.sqlite"
    if not source_db.is_file():
        raise FileNotFoundError(f"缺少01生成的原始行索引：{source_db}")
    crosswalk = {item["case_id"]: item["patient_id"] for item in read_jsonl(
        source_dir / "case_crosswalk.jsonl")}
    checked, seen = 0, set()
    with sqlite3.connect(source_db) as connection:
        for row in qgtk_rows:
            patient_id = crosswalk.get(row["case_id"])
            if not patient_id:
                raise ValueError(f"{row['case_id']} 缺少私有患者对照")
            for card in row["evidence_cards"]:
                source = card["source"]
                key = (patient_id, source["table"], source["csv_line"], source["row_sha256"])
                if key in seen:
                    continue
                found = connection.execute(
                    "SELECT 1 FROM records WHERE patient_id=? AND group_name=? AND csv_line=? AND row_sha=? LIMIT 1",
                    key).fetchone()
                if not found:
                    raise ValueError(f"{row['question_id']}/{card['evidence_id']} 原始行哈希或患者连接不符")
                seen.add(key)
                checked += 1
    return {"source_signature": signature, "source_rows_verified": checked,
            "source_index": str(source_db)}


def build_documents(qgtk_rows: list[dict], answers: dict[str, dict]) -> list[dict]:
    documents = []
    for row in qgtk_rows:
        qid = row["question_id"]
        dates = list(dict.fromkeys(card["date"] for card in row["evidence_cards"]))
        prefix = "记录日期：" + "、".join(dates) + "。"
        for role, answer in (("clean_doc", row["Ground Truth"]),
                             ("poisoned_doc", answers[qid]["Hallucinated Answer"])):
            body = prefix + answer.strip().rstrip("。") + "。"
            if row["case_id"] in body or re.search(r"clean_doc|poisoned_doc|投毒|错误答案|目标错误", body, re.I):
                raise ValueError(f"{qid} 的文档正文包含内部编号或实验标签")
            documents.append({"doc_id": f"{qid}:{role}", "question_id": qid,
                              "case_id": row["case_id"], "doc_role": role,
                              "is_poisoned": role == "poisoned_doc", "text": body,
                              "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                              "source_dates": dates, "synthetic_only": row["synthetic_only"]})
    return documents


def numeric(value: str) -> Decimal | None:
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    if not numbers:
        return None
    try:
        return Decimal(numbers[0])
    except InvalidOperation:
        return None


def build_trusted_kg(qgtk_rows: list[dict]) -> tuple[list[dict], list[dict], nx.MultiDiGraph, dict]:
    """Receives only trusted Q/GT/K rows and their verified original-row cards."""
    by_evidence, case_by_evidence, question_ids = {}, {}, defaultdict(set)
    for row in qgtk_rows:
        for card in row["evidence_cards"]:
            eid = card["evidence_id"]
            if eid in by_evidence and by_evidence[eid] != card:
                raise ValueError(f"证据ID对应不同原始事实：{eid}")
            if eid in case_by_evidence and case_by_evidence[eid] != row["case_id"]:
                raise ValueError(f"证据ID跨病例重复：{eid}")
            by_evidence[eid] = card
            case_by_evidence[eid] = row["case_id"]
            question_ids[eid].add(row["question_id"])
    graph = nx.MultiDiGraph()
    facts, slot_values = [], defaultdict(set)
    for eid, card in sorted(by_evidence.items()):
        case_id = case_by_evidence[eid]
        subject = f"{case_id}:{card['visit_token']}:{card['date']}"
        predicate = f"{card['table']}:{card['field']}"
        slot = digest([subject, card["table"], card["label"]])[:20]
        fact_id = f"F{eid}"
        fact = {"fact_id": fact_id, "slot_id": slot, "case_id": case_id,
                "subject": subject, "predicate": predicate, "object": card["required_value"],
                "label": card["label"], "record_date": card["date"],
                "date_basis": card["date_basis"], "visit_token": card["visit_token"],
                "evidence_id": eid, "question_ids": sorted(question_ids[eid]),
                "source": card["source"], "derivation": "source_row_and_grounded_QGTK"}
        facts.append(fact)
        slot_values[slot].add(card["required_value"])
        value_node = fact_id + ":value"
        graph.add_node(subject, node_type="case_visit", case_id=case_id)
        graph.add_node(value_node, node_type="trusted_value", value=card["required_value"],
                       label=card["label"], date=card["date"])
        graph.add_edge(subject, value_node, key=fact_id, fact_id=fact_id,
                       relation=predicate, source_row_sha256=card["source"]["row_sha256"])
    changes = []
    seen_change = set()
    for row in qgtk_rows:
        if row.get("task_type") != "paired_visit_comparison" or len(row["evidence_cards"]) != 2:
            continue
        first, second = sorted(row["evidence_cards"], key=lambda card: card["date"])
        if first["label"] != second["label"]:
            continue
        if first["table"] in {"诊断信息", "药品信息", "病史", "检查信息", "医嘱信息", "手术信息"}:
            continue
        before, after = numeric(first["required_value"]), numeric(second["required_value"])
        if before is None or after is None:
            continue
        change_id = "C" + digest([first["evidence_id"], second["evidence_id"]])[:20]
        if change_id in seen_change:
            continue
        seen_change.add(change_id)
        direction = "decreased" if after < before else "increased" if after > before else "unchanged"
        changes.append({"change_id": change_id, "case_id": row["case_id"],
                        "subject": f"{row['case_id']}:{first['label']}", "predicate": "change_direction",
                        "object": direction, "label": first["label"],
                        "before": first["required_value"], "after": second["required_value"],
                        "evidence_ids": [first["evidence_id"], second["evidence_id"]],
                        "question_ids": [row["question_id"]],
                        "derivation": "deterministic_comparison_of_source_values"})
        graph.add_edge(f"F{first['evidence_id']}:value", f"F{second['evidence_id']}:value",
                       key=change_id, change_id=change_id, relation="change_direction",
                       direction=direction)
    return facts, changes, graph, {"ambiguous_slots": sum(len(values) > 1 for values in slot_values.values())}


def build_sqlite(output_dir: Path, documents: list[dict], facts: list[dict], changes: list[dict]) -> None:
    kb_path, kg_path = output_dir / "kb.sqlite", output_dir / "trusted_kg.sqlite"
    for path in (kb_path, kg_path):
        if path.exists():
            path.unlink()
    with sqlite3.connect(kb_path) as connection:
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, question_id TEXT, case_id TEXT, "
                           "doc_role TEXT, text TEXT, text_sha256 TEXT)")
        connection.execute("CREATE INDEX documents_case_idx ON documents(case_id)")
        connection.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(doc_id UNINDEXED, text)")
        connection.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?)", [
            (doc["doc_id"], doc["question_id"], doc["case_id"], doc["doc_role"],
             doc["text"], doc["text_sha256"]) for doc in documents])
        connection.executemany("INSERT INTO documents_fts(doc_id,text) VALUES (?,?)",
                               [(doc["doc_id"], doc["text"]) for doc in documents])
    with sqlite3.connect(kg_path) as connection:
        connection.execute("CREATE TABLE facts (fact_id TEXT PRIMARY KEY, slot_id TEXT, case_id TEXT, "
                           "subject TEXT, predicate TEXT, object TEXT, label TEXT, record_date TEXT, "
                           "evidence_id TEXT, source_json TEXT, question_ids_json TEXT)")
        connection.execute("CREATE INDEX facts_case_slot ON facts(case_id,slot_id)")
        connection.execute("CREATE TABLE changes (change_id TEXT PRIMARY KEY, case_id TEXT, label TEXT, "
                           "direction TEXT, before_value TEXT, after_value TEXT, evidence_ids_json TEXT)")
        connection.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            (fact["fact_id"], fact["slot_id"], fact["case_id"], fact["subject"],
             fact["predicate"], fact["object"], fact["label"], fact["record_date"],
             fact["evidence_id"], json.dumps(fact["source"], ensure_ascii=False),
             json.dumps(fact["question_ids"], ensure_ascii=False)) for fact in facts])
        connection.executemany("INSERT INTO changes VALUES (?,?,?,?,?,?,?)", [
            (change["change_id"], change["case_id"], change["label"], change["object"],
             change["before"], change["after"], json.dumps(change["evidence_ids"])) for change in changes])


def build_indices(output_dir: Path, documents: list[dict]) -> None:
    for name, subset in (("clean", [doc for doc in documents if doc["doc_role"] == "clean_doc"]),
                         ("mixed", documents)):
        folder = output_dir / f"{name}_tfidf_index"
        folder.mkdir(parents=True, exist_ok=True)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                     sublinear_tf=True, norm="l2", max_features=200000)
        matrix = vectorizer.fit_transform([doc["text"] for doc in subset])
        save_npz(folder / "document_vectors.npz", matrix)
        joblib.dump(vectorizer, folder / "vectorizer.joblib")
        write_json(folder / "manifest.json", {"index_type": "local_character_ngram_tfidf",
                   "documents": len(subset), "doc_ids": [doc["doc_id"] for doc in subset],
                   "patient_scope_required": True, "model_visible_fields": ["text"],
                   "embedding_api_called": False})


def audit_outputs(output_dir: Path) -> dict:
    documents = read_jsonl(output_dir / "kb_documents.jsonl")
    facts = read_jsonl(output_dir / "trusted_kg_facts.jsonl")
    changes = read_jsonl(output_dir / "trusted_kg_changes.jsonl")
    target_coverage = read_jsonl(output_dir / "target_coverage.jsonl")
    roles = Counter(doc["doc_role"] for doc in documents)
    if not documents or roles["clean_doc"] != roles["poisoned_doc"] or sum(roles.values()) != len(documents):
        raise ValueError("clean/poison文档数量不相等")
    if len({doc["doc_id"] for doc in documents}) != len(documents):
        raise ValueError("KB文档ID重复")
    if any(doc["case_id"] in doc["text"] or "poisoned_doc" in doc["text"] or "clean_doc" in doc["text"]
           for doc in documents):
        raise ValueError("KB正文泄漏病例别名或角色")
    if len({fact["fact_id"] for fact in facts}) != len(facts):
        raise ValueError("KG事实ID重复")
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    if (len(target_coverage) != roles["clean_doc"] or
            any(item["trusted_fact_id"] not in facts_by_id or
                item["trusted_value"] != facts_by_id[item["trusted_fact_id"]]["object"] or
                item["trusted_value"] == item["replacement_value"] for item in target_coverage)):
        raise ValueError("目标错误与KG真值对应检查失败")
    with sqlite3.connect(output_dir / "kb.sqlite") as connection:
        kb_count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        fts_count = connection.execute("SELECT count(*) FROM documents_fts").fetchone()[0]
    with sqlite3.connect(output_dir / "trusted_kg.sqlite") as connection:
        fact_count = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
        change_count = connection.execute("SELECT count(*) FROM changes").fetchone()[0]
    if (kb_count, fts_count, fact_count, change_count) != (len(documents), len(documents), len(facts), len(changes)):
        raise ValueError("SQLite与JSONL行数不一致")
    for name, expected in (("clean", roles["clean_doc"]), ("mixed", len(documents))):
        folder = output_dir / f"{name}_tfidf_index"
        index = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        matrix = load_npz(folder / "document_vectors.npz")
        if index["documents"] != expected or matrix.shape[0] != expected or len(index["doc_ids"]) != expected:
            raise ValueError(f"{name} TF-IDF索引与文档数不符")
    return {"schema_version": SCHEMA, "questions": roles["clean_doc"],
            "documents": len(documents), "clean_documents": roles["clean_doc"],
            "poisoned_documents": roles["poisoned_doc"], "trusted_kg_facts": len(facts),
            "trusted_kg_changes": len(changes), "kg_reads_poison_answers": False,
            "target_source_facts_covered": len(target_coverage),
            "patient_scope_required": True, "embedding_api_called": False,
            "requires_manual_review": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "audit"), required=True)
    parser.add_argument("--qgtk", type=Path, default=QGTK)
    parser.add_argument("--answers", type=Path, default=ANSWERS)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "build":
        qgtk_rows, answers, pairing = paired_inputs(read_jsonl(args.qgtk), read_jsonl(args.answers))
        source_audit = verify_original_rows(qgtk_rows, args.source_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # KG has a deliberately separate function signature: it never receives answers.
        facts, changes, graph, kg_audit = build_trusted_kg(qgtk_rows)
        documents = build_documents(qgtk_rows, answers)
        facts_by_evidence = {fact["evidence_id"]: fact for fact in facts}
        target_coverage = []
        for row in qgtk_rows:
            target = answers[row["question_id"]]["target_plan"]
            fact = facts_by_evidence.get(target["evidence_id"])
            if fact:
                target_coverage.append({"question_id": row["question_id"],
                                        "target_evidence_id": target["evidence_id"],
                                        "trusted_fact_id": fact["fact_id"],
                                        "trusted_value": fact["object"],
                                        "replacement_value": answers[row["question_id"]]["replacement_value"]})
        if len(target_coverage) != len(qgtk_rows):
            raise ValueError("可信KG未覆盖全部错误答案的目标原始事实")
        write_jsonl(args.output_dir / "kb_documents.jsonl", documents)
        write_jsonl(args.output_dir / "trusted_kg_facts.jsonl", facts)
        write_jsonl(args.output_dir / "trusted_kg_changes.jsonl", changes)
        write_jsonl(args.output_dir / "target_coverage.jsonl", target_coverage)
        nx.write_graphml(graph, args.output_dir / "trusted_kg.graphml")
        build_sqlite(args.output_dir, documents, facts, changes)
        build_indices(args.output_dir, documents)
        write_json(args.output_dir / "build_manifest.json", {"schema_version": SCHEMA,
                   "qgtk_sha256": file_digest(args.qgtk), "answer_sha256": file_digest(args.answers),
                   "pairing": pairing, "source_audit": source_audit, "kg_audit": kg_audit,
                   "kg_input_fields": ["evidence_cards", "evidence_cards.source.row_sha256"],
                   "kg_grounding_gate_fields": ["Ground Truth", "Knowledge"],
                   "kg_forbidden_inputs": ["Hallucinated Answer", "target_plan", "doc_role", "RAG Answer"],
                   "target_source_facts_covered": len(target_coverage), "model_called": False})
    summary = audit_outputs(args.output_dir)
    write_json(args.output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
