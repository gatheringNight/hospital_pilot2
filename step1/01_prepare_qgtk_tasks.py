"""Prepare all Q/GT/K tasks from the 16-table hospital field schema.

This offline stage imports every one of the 244 field positions, joins records by
patient and encounter, classifies fields, extracts source-linked clinical facts,
and selects variable-size evidence-card groups. It never calls a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/field_schema_manifest.json"
CONFIG = ROOT / "config/table_mapping.example.json"
DATA = ROOT / "private/exports"
OUTPUT = ROOT / "private/step1_output"
SCHEMA = "hospital-qgtk-task-v2"
IDENTIFIERS = {"EMPIID", "patid", "就诊流水号", "住院门诊号", "住院号", "报告单号", "报告号", "检查号", "患者标记"}
CONTEXT = {"基线或随访", "门诊住院标志", "就诊类型", "检验项目名称", "检验英文名称", "申请项目",
           "检验类型", "单位", "检查类型", "药品规格", "剂量单位", "剂型名称", "检验时间", "报告时间",
           "录入日期", "检查日期", "手术日期", "入院日期", "出院日期", "医嘱执行时间", "医嘱结束时间"}
ADMIN = {"入院科室名称", "出院科室名称", "入院病区名称", "出院病区名称", "科室编码", "科室名称",
         "病区编码", "病区名称", "特色门诊标志", "医保类型", "医保说明", "民族", "籍贯",
         "婚姻状况", "患者类型", "是否二次入院", "就诊次数", "自备药标志"}
DATE_BY_GROUP = {"护理信息": "录入日期", "检验信息": "检验时间", "检查信息": "检查时间",
                 "医嘱信息": "医嘱执行时间", "药品信息": "医嘱执行时间", "动态血压": "检查日期",
                 "动态心电图": "检查日期", "手术信息": "手术日期", "脉搏波速度": "检查日期",
                 "心超": "检查日期", "颈动脉超声": "检查日期"}


def hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def field_role(group: str, field: str) -> str:
    if field in IDENTIFIERS:
        return "identifier"
    if group == "人口学信息表":
        return "patient_context_only"
    if field in ADMIN or field.endswith("编码") or field.endswith("代码"):
        return "administrative_or_code"
    if field in CONTEXT or field.endswith("日期") or field.endswith("时间"):
        return "event_context"
    return "clinical_candidate"


def manifest_and_config(args) -> tuple[list[dict], dict, dict]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = manifest["groups"]
    if len(groups) != 16 or sum(len(group["fields"]) for group in groups) != 244:
        raise ValueError("字段清单不是16组、244个字段位置")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    files = {group["name"]: args.data_dir / config.get("files", {}).get(
        group["name"], config.get("file_pattern", "{group}.csv").format(group=group["name"]))
             for group in groups}
    if not files["就诊信息"].is_file():
        raise FileNotFoundError("缺少必需的就诊信息.csv；其余表组允许缺失并会记录覆盖缺口")
    return groups, config, files


def index_exports(db_path: Path, groups: list[dict], config: dict, files: dict[str, Path]) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS records (group_name TEXT, patient_id TEXT, encounter_id TEXT, "
                       "csv_line INTEGER, row_sha TEXT, row_json TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS import_stats (group_name TEXT PRIMARY KEY, rows_total INTEGER, "
                       "rows_with_patient INTEGER, field_nonempty_json TEXT, missing_columns_json TEXT)")
    imported = {row[0] for row in connection.execute("SELECT group_name FROM import_stats")}
    aliases = config.get("field_aliases", {})
    patient_key = config.get("patient_key", "EMPIID")
    encounter_key = config.get("encounter_key", "就诊流水号")
    for group in groups:
        name = group["name"]
        if name in imported:
            continue
        source_path = files[name]
        mapping = {field: aliases.get(name, {}).get(field, field) for field in group["fields"]}
        if not source_path.is_file():
            connection.execute("INSERT INTO import_stats VALUES (?,?,?,?,?)",
                               (name, 0, 0, "{}", json.dumps(group["fields"], ensure_ascii=False)))
            connection.commit()
            continue
        with source_path.open("r", encoding=config.get("encoding", "utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError(f"{name}有重复表头")
            missing = [field for field, actual in mapping.items() if actual not in headers]
            required = {patient_key, encounter_key} if name != "人口学信息表" else {patient_key}
            if name == "就诊信息":
                required |= {config.get("visit_date", "入院日期")}
            if required & set(missing):
                raise ValueError(f"{name}缺少关联所必需字段：{sorted(required & set(missing))}")
            total, with_patient = 0, 0
            nonempty = Counter()
            buffer = []
            for line, raw in enumerate(reader, 2):
                if None in raw:
                    raise ValueError(f"{name}第{line}行数据多于表头")
                row = {field: (raw.get(actual) or "").strip() for field, actual in mapping.items()}
                total += 1
                nonempty.update(field for field, value in row.items() if value)
                patient_id = row.get(patient_key, "")
                if not patient_id:
                    continue
                with_patient += 1
                buffer.append((name, patient_id, row.get(encounter_key, ""), line,
                               hash_json(row), json.dumps(row, ensure_ascii=False)))
                if len(buffer) >= 1000:
                    connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", buffer)
                    buffer.clear()
            if buffer:
                connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", buffer)
            connection.execute("INSERT INTO import_stats VALUES (?,?,?,?,?)",
                               (name, total, with_patient, json.dumps(nonempty, ensure_ascii=False),
                                json.dumps(missing, ensure_ascii=False)))
            connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS idx_patient ON records(patient_id)")
    connection.commit()
    stats = {name: {"rows_total": total, "rows_with_patient": with_patient,
                    "field_nonempty": json.loads(counts), "missing_columns": json.loads(missing)}
             for name, total, with_patient, counts, missing in connection.execute("SELECT * FROM import_stats")}
    connection.close()
    return stats


def patient_rows(connection: sqlite3.Connection, patient_id: str) -> list[dict]:
    result = []
    for group, encounter, line, row_sha, encoded in connection.execute(
            "SELECT group_name, encounter_id, csv_line, row_sha, row_json FROM records "
            "WHERE patient_id=? ORDER BY group_name, csv_line", (patient_id,)):
        result.append({"group": group, "encounter": encounter, "csv_line": line,
                       "row_sha256": row_sha, "values": json.loads(encoded)})
    return result


def visit_date(value: str) -> str:
    text = value.strip().replace("/", "-")[:10]
    return date.fromisoformat(text).isoformat()


def visits_for_patient(rows: list[dict], config: dict) -> dict[str, dict]:
    visit_rows = [row for row in rows if row["group"] == "就诊信息"]
    visits = {}
    for row in visit_rows:
        encounter = row["encounter"]
        if not encounter:
            continue
        values = row["values"]
        try:
            when = visit_date(values.get(config.get("visit_date", "入院日期"), ""))
        except ValueError:
            continue
        label = values.get(config.get("visit_role", "基线或随访"), "").strip()
        current = {"encounter": encounter, "date": when, "label": label,
                   "source_row_sha256": row["row_sha256"]}
        if encounter in visits and (visits[encounter]["date"] != when or visits[encounter]["label"] != label):
            raise ValueError("同一就诊流水号对应不同日期或基线/随访标签")
        visits[encounter] = current
    if not visits:
        raise ValueError("没有可连接的就诊记录")
    return visits


def text_value(value: str) -> bool:
    return bool(value and value.strip() and len(value.strip()) <= 500 and
                value.strip().lower() not in {"null", "nan", "不详", "未测", "未做"})


def fact_label(group: str, field: str, values: dict) -> str:
    if group == "检验信息":
        item = values.get("检验项目名称") or values.get("申请项目") or "该检验"
        return f"{item}的{field}"
    if group == "药品信息":
        item = values.get("药品名称") or "该药品"
        return f"{item}的{field}" if field != "药品名称" else "药品名称"
    if group == "检查信息":
        return f"{values.get('检查类型') or '该检查'}的{field}"
    return field


def fact_cards(case_id: str, rows: list[dict], visits: dict[str, dict], config: dict) -> tuple[list[dict], Counter]:
    cards, skipped = [], Counter()
    private_tokens = {value for record in rows for field, value in record["values"].items()
                      if field in IDENTIFIERS and len(value) >= 6}
    for record in rows:
        group, encounter, values = record["group"], record["encounter"], record["values"]
        if group in {"人口学信息表", "就诊信息"}:
            continue
        visit = visits.get(encounter)
        if not visit:
            skipped["unlinked_encounter"] += 1
            continue
        time_field = DATE_BY_GROUP.get(group)
        raw_time = values.get(time_field, "") if time_field else ""
        try:
            when = visit_date(raw_time) if raw_time else visit["date"]
        except ValueError:
            skipped["invalid_event_time"] += 1
            continue
        for field, value in values.items():
            if field_role(group, field) != "clinical_candidate" or not text_value(value):
                if field_role(group, field) == "clinical_candidate" and value and len(value.strip()) > 500:
                    skipped["long_clinical_text"] += 1
                continue
            label = fact_label(group, field, values)
            display = value
            if group == "检验信息" and field == "数值型结果" and values.get("单位"):
                display = f"{value}{values['单位']}" if values["单位"] == "%" else f"{value} {values['单位']}"
            if group == "药品信息" and field in {"药品剂量", "药品单次剂量", "药品日剂量", "药品总剂量"} and values.get("剂量单位"):
                display = f"{value} {values['剂量单位']}"
            if any(token in display or token in label for token in private_tokens):
                skipped["identifier_inside_clinical_text"] += 1
                continue
            evidence_id = "E" + hash_json([case_id, group, encounter, record["row_sha256"], field])[:12]
            source_fields = [field]
            if time_field:
                source_fields.append(time_field)
            if group == "检验信息":
                source_fields.extend(["检验项目名称", "单位"])
            if group == "药品信息":
                source_fields.extend(["药品名称", "剂量单位"])
            if group == "检查信息":
                source_fields.append("检查类型")
            cards.append({"evidence_id": evidence_id, "table": group, "field": field,
                          "label": label, "required_value": display,
                          "fact": f"{label}：{display}", "date": when,
                          "date_basis": "event_field" if raw_time else "linked_visit_date",
                          "visit_label": visit["label"] or "该次就诊",
                          "visit_token": "V" + hash_json(encounter)[:10],
                          "source": {"table": group, "csv_line": record["csv_line"],
                                     "fields": list(dict.fromkeys(source_fields)),
                                     "row_sha256": record["row_sha256"]}})
    return cards, skipped


def make_tasks(case_id: str, cards: list[dict], config: dict, source_signature: str,
               synthetic: bool) -> list[dict]:
    max_cards = int(config.get("max_cards_per_question", 6))
    if max_cards < 2 or max_cards > 10:
        raise ValueError("max_cards_per_question须为2至10")
    groups = defaultdict(list)
    for card in cards:
        groups[(card["table"], card["visit_token"], card["date"])].append(card)
    tasks = []
    for (table, visit_token, record_date), selected in sorted(groups.items()):
        selected.sort(key=lambda card: (card["source"]["csv_line"], card["field"]))
        for offset in range(0, len(selected), max_cards):
            chunk = selected[offset:offset + max_cards]
            if len(chunk) < 2:
                continue
            labels = list(dict.fromkeys(card["label"] for card in chunk))
            keywords = list(dict.fromkeys([table.replace("信息", "").replace("表", "")] +
                                           [label.split("的", 1)[0] for label in labels]))
            task_key = [case_id, table, visit_token, record_date, offset,
                        [card["evidence_id"] for card in chunk]]
            task_id = "Q" + hash_json(task_key)[:14]
            tasks.append({"schema_version": SCHEMA, "question_id": task_id, "case_id": case_id,
                          "synthetic_only": synthetic, "source_signature": source_signature,
                          "task_type": "single_visit_records", "topic": table,
                          "question_focus": labels, "question_keywords": keywords,
                          "question_intent": f"询问该患者{chunk[0]['date']}的{table}记录中相关事实",
                          "evidence_cards": chunk,
                          "knowledge_groups": [[card["evidence_id"] for card in chunk[i:i + 3]]
                                               for i in range(0, len(chunk), 3)],
                          "required_claims": [{"evidence_id": card["evidence_id"], "text": card["fact"]}
                                              for card in chunk]})
    # Paired questions use identical table/field/item labels across two dated encounters.
    paired = defaultdict(list)
    for card in cards:
        paired[(card["table"], card["label"])].append(card)
    comparable_tables = {"检验信息", "护理信息", "诊断信息", "药品信息", "动态血糖", "动态血压", "病史"}
    for (table, label), candidates in sorted(paired.items()):
        if table not in comparable_tables:
            continue
        by_visit = defaultdict(list)
        for card in candidates:
            by_visit[card["visit_token"]].append(card)
        if len(by_visit) != 2 or any(len(part) != 1 for part in by_visit.values()):
            continue
        pair = sorted((part[0] for part in by_visit.values()), key=lambda card: card["date"])
        if pair[0]["date"] == pair[1]["date"] or pair[0]["required_value"] == pair[1]["required_value"]:
            continue
        task_id = "Q" + hash_json([case_id, "paired", table, label, [c["evidence_id"] for c in pair]])[:14]
        tasks.append({"schema_version": SCHEMA, "question_id": task_id, "case_id": case_id,
                      "synthetic_only": synthetic, "source_signature": source_signature,
                      "task_type": "paired_visit_comparison", "topic": table,
                      "question_focus": [label],
                      "question_keywords": [table.replace("信息", "").replace("表", ""), label.split("的", 1)[0]],
                      "question_intent": f"对照该患者{pair[0]['date']}与{pair[1]['date']}两次{table}记录，询问{label}的变化",
                      "evidence_cards": pair,
                      "knowledge_groups": [[pair[0]["evidence_id"]], [pair[1]["evidence_id"]]],
                      "required_claims": [{"evidence_id": card["evidence_id"], "text": card["fact"]}
                                          for card in pair]})
    # Preserve broad table coverage without sending every nonempty field to a 3B model.
    limit = int(config.get("max_questions_per_patient", 12))
    if limit < 0:
        raise ValueError("max_questions_per_patient不能为负数")
    if not limit or len(tasks) <= limit:
        return sorted(tasks, key=lambda task: task["question_id"])
    core = ["检验信息", "护理信息", "诊断信息", "药品信息", "病史", "检查信息"]
    other = ["动态血糖", "动态血压", "动态心电图", "医嘱信息", "手术信息", "脉搏波速度",
             "心超", "颈动脉超声"]
    rotation = int(hash_json(case_id)[:4], 16) % len(other)
    table_order = core + other[rotation:] + other[:rotation]
    paired_options = [task for task in tasks if task["task_type"] == "paired_visit_comparison"]
    paired_options.sort(key=lambda task: (
        task["topic"] not in {"检验信息", "护理信息"},
        task["topic"] not in core, task["question_id"]))
    chosen = paired_options[:min(2, limit)]
    for table in table_order:
        options = [task for task in tasks if task["topic"] == table and
                   task["task_type"] == "single_visit_records"]
        if options:
            chosen.append(max(options, key=lambda task: (len(task["evidence_cards"]), task["question_id"])))
        if len(chosen) >= limit:
            break
    # Reserve remaining slots for changed-value comparisons and uncovered facts.
    remaining = [task for task in tasks if task["question_id"] not in {item["question_id"] for item in chosen}]
    remaining.sort(key=lambda task: (
        task["task_type"] == "paired_visit_comparison",
        task["topic"] in core,
        len(task["evidence_cards"]), task["question_id"]), reverse=True)
    chosen += remaining[:max(0, limit - len(chosen))]
    return sorted(chosen, key=lambda task: task["question_id"])


def get_salt(output_dir: Path) -> bytes:
    path = output_dir / "case_id_salt.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="ascii")
    return bytes.fromhex(path.read_text(encoding="ascii").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-patients", type=int, default=0, help="0为所有患者；正数仅用于院内小批预检")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="仅用于公开的虚构测试CSV")
    args = parser.parse_args()
    if args.max_patients < 0:
        raise ValueError("--max-patients不能为负数")
    groups, config, files = manifest_and_config(args)
    signatures = {name: hash_file(path) if path.is_file() else "MISSING" for name, path in files.items()}
    source_signature = hash_json({"preparer_version": "tolerant-v3", "manifest": hash_file(args.manifest), "config": hash_file(args.config),
                                  "files": signatures})
    index_path = args.output_dir / f"source_index_{source_signature[:12]}.sqlite"
    stats = index_exports(index_path, groups, config, files)
    salt = get_salt(args.output_dir)
    work_dir = args.output_dir / "work" / source_signature[:12]
    work_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    patients = [row[0] for row in connection.execute(
        "SELECT DISTINCT patient_id FROM records WHERE group_name='就诊信息' ORDER BY patient_id")]
    selected_patients = patients[:args.max_patients] if args.max_patients else patients
    cases_processed = 0
    for patient_id in selected_patients:
        case_id = ("DEMO" if args.synthetic else "HOSP") + "-" + hmac.new(
            salt, patient_id.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
        case_file = work_dir / f"{case_id}.json"
        if case_file.exists() and args.resume:
            cases_processed += 1
            continue
        try:
            rows = patient_rows(connection, patient_id)
            visits = visits_for_patient(rows, config)
            cards, skipped = fact_cards(case_id, rows, visits, config)
            tasks = make_tasks(case_id, cards, config, source_signature, args.synthetic)
            payload = {"case_id": case_id, "status": "prepared" if tasks else "no_eligible_tasks",
                       "tasks": tasks, "cards_created": len(cards), "skipped_rows": dict(skipped),
                       "patient_id_for_private_crosswalk": patient_id,
                       "encounter_ids_for_private_crosswalk": sorted(visits)}
        except (ValueError, KeyError) as error:
            payload = {"case_id": case_id, "status": "excluded", "tasks": [],
                       "reason": str(error), "patient_id_for_private_crosswalk": patient_id}
        write_json(case_file, payload)
        cases_processed += 1
        if cases_processed % 100 == 0:
            print(f"patients prepared: {cases_processed}/{len(selected_patients)}", flush=True)
    connection.close()
    all_case_files = sorted(work_dir.glob("*.json"))
    task_path = args.output_dir / "qgtk_tasks.jsonl"
    crosswalk_path = args.output_dir / "case_crosswalk.jsonl"
    exclusions_path = args.output_dir / "excluded_cases.jsonl"
    field_selection = Counter()
    task_types = Counter()
    task_count = card_uses = patient_with_tasks = excluded_count = 0
    task_ids = set()
    with task_path.open("w", encoding="utf-8") as task_handle, \
         crosswalk_path.open("w", encoding="utf-8") as crosswalk_handle, \
         exclusions_path.open("w", encoding="utf-8") as excluded_handle:
        for path in all_case_files:
            case = json.loads(path.read_text(encoding="utf-8"))
            crosswalk_handle.write(json.dumps({"case_id": case["case_id"],
                "patient_id": case["patient_id_for_private_crosswalk"],
                "encounter_ids": case.get("encounter_ids_for_private_crosswalk", [])}, ensure_ascii=False) + "\n")
            if case["status"] == "prepared":
                patient_with_tasks += 1
            else:
                excluded_count += 1
                excluded_handle.write(json.dumps({"case_id": case["case_id"],
                    "status": case["status"], "reason": case.get("reason")}, ensure_ascii=False) + "\n")
            for task in case.get("tasks", []):
                if task["question_id"] in task_ids:
                    raise ValueError("问题ID碰撞或重复")
                task_ids.add(task["question_id"])
                task_handle.write(json.dumps(task, ensure_ascii=False) + "\n")
                task_count += 1
                task_types[task["task_type"]] += 1
                card_uses += len(task["evidence_cards"])
                field_selection.update((card["table"], card["field"]) for card in task["evidence_cards"])
    field_coverage = [{"table": group["name"], "field": field,
                       "role": field_role(group["name"], field),
                       "source_column_present": field not in stats[group["name"]]["missing_columns"],
                       "nonempty_rows": stats[group["name"]]["field_nonempty"].get(field, 0),
                       "selected_card_count": field_selection[(group["name"], field)]}
                      for group in groups for field in group["fields"]]
    write_json(args.output_dir / "field_coverage_244.json", field_coverage)
    summary = {"schema_version": SCHEMA, "source_signature": source_signature,
               "table_groups": len(groups), "field_positions": len(field_coverage),
               "missing_table_exports": [name for name, path in files.items() if not path.is_file()],
               "missing_field_positions_in_exports": sum(len(data["missing_columns"]) for data in stats.values()),
               "indexed_rows_by_table": {name: data["rows_total"] for name, data in stats.items()},
               "patients_in_visit_table": len(patients), "patients_processed": len(all_case_files),
               "patients_with_tasks": patient_with_tasks,
               "tasks": task_count, "task_types": dict(task_types),
               "evidence_card_uses": card_uses,
               "nonempty_clinical_field_positions": sum(item["role"] == "clinical_candidate" and item["nonempty_rows"] > 0
                                                       for item in field_coverage),
               "clinical_field_positions_selected": sum(item["role"] == "clinical_candidate" and item["selected_card_count"] > 0
                                                        for item in field_coverage),
               "excluded_cases": excluded_count,
               "synthetic_only": args.synthetic, "model_called": False,
               "output": str(args.output_dir / "qgtk_tasks.jsonl")}
    write_json(args.output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if not task_count:
        raise RuntimeError("没有合格任务；检查 excluded_cases.jsonl 和 field_coverage_244.json")


if __name__ == "__main__":
    main()
