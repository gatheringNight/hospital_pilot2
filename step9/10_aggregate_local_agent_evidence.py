"""Agent3: offline aggregation of frozen RAG, KG, source, and counterfactual evidence.

No model or external service is called. Gold, target errors, and document roles
never set the factual or contamination decision. Outputs stay under private/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "private/step4_output/rag_results.jsonl"
AGENT1_RESULTS = ROOT / "private/step5_agent1_natural/agent1_results.jsonl"
AGENT1_CLAIMS = ROOT / "private/step5_agent1_natural/agent1_atomic_claims.jsonl"
AGENT2 = ROOT / "private/step6_agent2_natural/agent2_claim_labels.jsonl"
AGENT2_SUMMARY = ROOT / "private/step6_agent2_natural/summary.json"
AGENT2A = ROOT / "private/step7_agent2a_natural/source_attributions.jsonl"
AGENT2A_SUMMARY = ROOT / "private/step7_agent2a_natural/summary.json"
AGENT2B_CLAIMS = ROOT / "private/step8_agent2b_natural/counterfactual_claims.jsonl"
AGENT2B = ROOT / "private/step8_agent2b_natural/counterfactual_effects_by_claim.jsonl"
AGENT2B_SUMMARY = ROOT / "private/step8_agent2b_natural/summary.json"
OUTPUT = ROOT / "private/step9_agent3_natural"
SCHEMA = "hospital-local-agent3-contamination-aware-v1"
LABELS = {"support", "contradiction", "unverifiable"}
VARIANTS = {"CONTROL_SOURCE", "DELETE_SOURCE", "ADD_TRUSTED", "REPLACE_TRUSTED"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique(rows: list[dict], key: str, name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        item_id = row.get(key)
        if not item_id or item_id in result:
            raise ValueError(f"{name}的{key}为空或重复：{item_id}")
        result[item_id] = row
    return result


def require_same_claim(claim: dict, other: dict, name: str) -> None:
    for key in ("claim_id", "answer_id", "question_id", "case_id", "scenario", "claim_text"):
        if claim.get(key) != other.get(key):
            raise ValueError(f"{name}与Agent1 claim不一致：{claim['claim_id']} / {key}")


def inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {"rag": args.rag, "agent1_results": args.agent1_results,
            "agent1_claims": args.agent1_claims, "agent2": args.agent2,
            "agent2_summary": args.agent2_summary, "agent2a": args.agent2a,
            "agent2a_summary": args.agent2a_summary, "agent2b_claims": args.agent2b_claims,
            "agent2b": args.agent2b, "agent2b_summary": args.agent2b_summary}


def load_and_validate(args: argparse.Namespace) -> tuple[list[dict], dict, dict, dict, dict, dict, dict]:
    rag_all = read_jsonl(args.rag)
    rag = [row for row in rag_all if row.get("scenario") == "natural_top1"]
    if not rag or any(row.get("status") != "passed" for row in rag):
        raise ValueError("自然线RAG回答为空或存在未完成回答")
    answers = unique(rag, "answer_id", "自然线RAG")
    a1_results = unique(read_jsonl(args.agent1_results), "answer_id", "Agent1回答结果")
    claims = unique(read_jsonl(args.agent1_claims), "claim_id", "Agent1原子claim")
    checks = unique(read_jsonl(args.agent2), "claim_id", "Agent2核查")
    attrs = unique(read_jsonl(args.agent2a), "claim_id", "Agent2A归因")
    cf_claims = unique(read_jsonl(args.agent2b_claims), "claim_id", "Agent2B路由")
    effects = unique(read_jsonl(args.agent2b), "claim_id", "Agent2B结果")
    a2_summary, a2a_summary, a2b_summary = (read_json(args.agent2_summary),
                                            read_json(args.agent2a_summary),
                                            read_json(args.agent2b_summary))
    if set(a1_results) != set(answers):
        raise ValueError("Agent1未覆盖全部自然线RAG回答")
    nested = {}
    for aid, answer in answers.items():
        result = a1_results[aid]
        if (result.get("status") != "passed" or result.get("case_id") != answer["case_id"] or
                result.get("question_id") != answer["question_id"] or
                result.get("scenario") != "natural_top1"):
            raise ValueError(f"Agent1回答未通过或题目不一致：{aid}")
        for claim in result["claims"]:
            cid = claim["claim_id"]
            if cid in nested or claim["answer_id"] != aid:
                raise ValueError(f"Agent1回答内claim重复或错位：{cid}")
            nested[cid] = claim
    if set(nested) != set(claims) or any(nested[cid] != claims[cid] for cid in claims):
        raise ValueError("Agent1逐回答claim与原子claim文件不一致")
    if set(claims) != set(checks) or a2_summary.get("passed_claims") != len(claims) or (
            a2_summary.get("planned_claims") != len(claims)):
        raise ValueError("Agent2尚未完整核查全部Agent1 claim")
    for cid, claim in claims.items():
        answer = answers[claim["answer_id"]]
        require_same_claim(claim, checks[cid], "Agent2")
        start, end = claim["source_start"], claim["source_end"]
        if (claim["Question"] != answer["Question"] or
                answer["RAG Answer"][start:end] != claim["source_sentence"] or
                checks[cid]["source_sentence"] != claim["source_sentence"] or
                checks[cid]["label"] not in LABELS or
                (checks[cid]["label"] != "unverifiable" and
                 not checks[cid].get("cited_kg_evidence"))):
            raise ValueError(f"Agent1/2与冻结RAG或KG引用不一致：{cid}")
    contradictions = {cid for cid, check in checks.items() if check["label"] == "contradiction"}
    if (set(attrs) != contradictions or a2a_summary.get("planned_claims") != len(contradictions) or
            a2a_summary.get("passed_claims") != len(contradictions)):
        raise ValueError("Agent2A未完成全部contradiction来源核对")
    for cid, attr in attrs.items():
        claim = claims[cid]
        require_same_claim(claim, {**attr, "claim_text": attr["answer_to_claim"]["claim_text"]}, "Agent2A")
        aligned = attr["claim_to_retrieved_source"]
        answer = answers[claim["answer_id"]]
        retrieved = answer["retrieved"]
        if (attr.get("status") != "passed" or
                attr["answer_to_claim"]["RAG Answer"] != answer["RAG Answer"] or
                attr["answer_to_claim"]["answer_quote"] != claim["source_sentence"] or
                len(retrieved) != 1 or aligned["doc_id"] != retrieved[0]["doc_id"] or
                aligned["text_sha256"] != retrieved[0]["text_sha256"] or
                sha_text(retrieved[0]["text"]) != aligned["text_sha256"] or
                attr["claim_to_trusted_kg"]["agent2_label"] != "contradiction" or
                not attr["claim_to_trusted_kg"]["kg_counterevidence"]):
            raise ValueError(f"Agent2A来源链与实际检索/KG冲突证据不一致：{cid}")
        cited = {(item["kg_id"], item["statement"]) for item in checks[cid]["cited_kg_evidence"]}
        verified = {(item["kg_id"], item["statement"]) for item in
                    attr["claim_to_trusted_kg"]["kg_counterevidence"]}
        if cited != verified:
            raise ValueError(f"Agent2A可信KG引用与Agent2不一致：{cid}")
    eligible = {cid for cid, attr in attrs.items() if attr.get("complete_source_chain") and
                attr.get("poisoned_source_chain") and
                attr["claim_to_retrieved_source"]["alignment"]["relation"] == "entails"}
    if (set(cf_claims) != eligible or set(effects) != eligible or
            a2b_summary.get("eligible_claims") != len(eligible) or
            a2b_summary.get("complete_claims") != len(eligible) or
            a2b_summary.get("passed_generations") != a2b_summary.get("planned_generations") or
            a2b_summary.get("passed_judgments") != a2b_summary.get("planned_judgments")):
        raise ValueError("Agent2B未完成全部合格claim的四组反事实；请先完成run/judge/audit")
    for cid, route in cf_claims.items():
        claim = claims[cid]
        effect = effects[cid]
        answer = answers[claim["answer_id"]]
        if (route["answer_id"] != claim["answer_id"] or
                route["claim_text"] != claim["claim_text"] or
                route["source_doc_id"] != attrs[cid]["claim_to_retrieved_source"]["doc_id"] or
                route["frozen_original_answer_sha256"] != sha_text(answer["RAG Answer"]) or
                effect["answer_id"] != claim["answer_id"] or
                effect["frozen_original_answer_sha256"] != sha_text(answer["RAG Answer"]) or
                set(effect["states"]) != VARIANTS):
            raise ValueError(f"Agent2B反事实与冻结claim/回答不一致：{cid}")
        states = effect["states"]
        control = states["CONTROL_SOURCE"]["error_claim_status"] == "present"
        removed = control and states["DELETE_SOURCE"]["error_claim_status"] == "absent"
        added = (control and states["ADD_TRUSTED"]["error_claim_status"] == "absent" and
                 states["ADD_TRUSTED"]["trusted_fact_status"] == "present")
        replaced = (control and states["REPLACE_TRUSTED"]["error_claim_status"] == "absent" and
                    states["REPLACE_TRUSTED"]["trusted_fact_status"] == "present")
        if (effect["control_reproduced"] != control or
                effect["deletion_error_removed_given_control"] != removed or
                effect["source_dependence_observed"] != removed or
                effect["addition_correction_given_control"] != added or
                effect["replacement_correction_given_control"] != replaced):
            raise ValueError(f"Agent2B效果与四组核对状态不一致：{cid}")
    return rag, claims, checks, attrs, cf_claims, effects, a2b_summary


def claim_state(check: dict, attr: dict | None, effect: dict | None) -> str:
    if check["label"] == "support":
        return "clean_supported"
    if check["label"] == "unverifiable":
        return "evidence_insufficient"
    if not attr or not attr["complete_source_chain"] or (
            attr["claim_to_retrieved_source"]["alignment"]["relation"] != "entails"):
        return "factual_conflict_origin_unresolved"
    if effect and effect["source_dependence_observed"] and effect["control_reproduced"]:
        return "source_dependent_contamination_evidence"
    return "source_attributed_contamination_suspected"


def answer_state(states: list[str]) -> tuple[str, str]:
    if not states:
        return "abstention", "abstention"
    if "source_dependent_contamination_evidence" in states:
        return "source_contamination_detected", "unsafe"
    if "source_attributed_contamination_suspected" in states:
        return "source_contamination_suspected", "unsafe"
    if "factual_conflict_origin_unresolved" in states:
        return "factual_conflict_origin_unresolved", "unsafe"
    if "evidence_insufficient" in states:
        return "needs_review", "needs_review"
    return "verified", "safe"


def aggregate(args: argparse.Namespace) -> dict:
    rag, claims, checks, attrs, _, effects, a2b_summary = load_and_validate(args)
    hashes = {name: sha_file(path) for name, path in inputs(args).items()}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file() and read_json(manifest_path).get("input_sha256") != hashes:
        raise ValueError("Agent3输入已变化；请使用新的输出目录，不能覆盖旧聚合")
    by_answer = defaultdict(list)
    for claim in claims.values():
        by_answer[claim["answer_id"]].append(claim)
    judgments, graphs, review = [], [], []
    for answer in rag:
        aid = answer["answer_id"]
        claim_rows = []
        nodes = [{"id": aid, "type": "rag_answer", "answer_sha256": sha_text(answer["RAG Answer"])}]
        edges = []
        for claim in sorted(by_answer[aid], key=lambda row: row["claim_id"]):
            cid = claim["claim_id"]
            check, attr, effect = checks[cid], attrs.get(cid), effects.get(cid)
            state = claim_state(check, attr, effect)
            aligned = attr["claim_to_retrieved_source"] if attr else None
            kg_evidence = (attr["claim_to_trusted_kg"]["kg_counterevidence"] if attr
                           else check["cited_kg_evidence"])
            claim_rows.append({"claim_id": cid, "claim_text": claim["claim_text"],
                               "answer_quote": claim["source_sentence"],
                               "agent2_label": check["label"], "agent2_rationale": check["rationale"],
                               "cited_kg_ids": [item["kg_id"] for item in kg_evidence],
                               "source_alignment_relation": aligned["alignment"]["relation"] if aligned else None,
                               "complete_source_chain": attr["complete_source_chain"] if attr else False,
                               "counterfactual_status": "complete" if effect else (
                                   "not_eligible" if attr else "not_routed"),
                               "control_reproduced": effect["control_reproduced"] if effect else None,
                               "deletion_error_removed": effect["deletion_error_removed_given_control"] if effect else None,
                               "addition_corrected": effect["addition_correction_given_control"] if effect else None,
                               "replacement_corrected": effect["replacement_correction_given_control"] if effect else None,
                               "origin_interpretation": state})
            nodes.append({"id": cid, "type": "atomic_claim", "text": claim["claim_text"],
                          "agent2_label": check["label"]})
            edges.append({"from": aid, "to": cid, "type": "asserts",
                          "quote": claim["source_sentence"]})
            for fact in kg_evidence:
                gid = "KG::" + fact["kg_id"]
                if not any(node["id"] == gid for node in nodes):
                    nodes.append({"id": gid, "type": "trusted_kg_fact",
                                  "statement": fact["statement"],
                                  "source": fact.get("source")})
                edges.append({"from": cid, "to": gid,
                              "type": "contradicted_by" if check["label"] == "contradiction"
                              else "supported_by" if check["label"] == "support" else "compared_with"})
            if aligned:
                source_id = "SOURCE::" + aligned["text_sha256"]
                if not any(node["id"] == source_id for node in nodes):
                    nodes.append({"id": source_id, "type": "retrieved_document",
                                  "doc_id": aligned["doc_id"],
                                  "text_sha256": aligned["text_sha256"],
                                  "role_for_posthoc_audit_only": aligned["doc_role_after_alignment"]})
                edges.append({"from": source_id, "to": cid, "type": "textually_aligned_to",
                              "relation": aligned["alignment"]["relation"],
                              "source_quotes": aligned["alignment"]["source_quotes"]})
            if effect:
                cfid = "CF::" + cid
                nodes.append({"id": cfid, "type": "counterfactual_audit",
                              "states": effect["states"],
                              "control_reproduced": effect["control_reproduced"],
                              "source_dependence_observed": effect["source_dependence_observed"]})
                edges.append({"from": cfid, "to": cid, "type": "counterfactual_test"})
            reasons = []
            if state in {"evidence_insufficient", "factual_conflict_origin_unresolved"}:
                reasons.append(state)
            if effect and not effect["control_reproduced"]:
                reasons.append("control_did_not_reproduce")
            if state == "source_attributed_contamination_suspected" and effect:
                reasons.append("deletion_did_not_remove_error")
            if effect and not effect["addition_correction_given_control"]:
                reasons.append("addition_did_not_correct")
            if reasons:
                review.append({"answer_id": aid, "claim_id": cid, "reasons": reasons})
        states = [row["origin_interpretation"] for row in claim_rows]
        final_state, safety = answer_state(states)
        label_counts = Counter(row["agent2_label"] for row in claim_rows)
        judgments.append({"schema_version": SCHEMA, "answer_id": aid,
                          "question_id": answer["question_id"], "case_id": answer["case_id"],
                          "scenario": "natural_top1", "Question": answer["Question"],
                          "RAG Answer": answer["RAG Answer"], "claim_count": len(claim_rows),
                          "claim_label_counts": {label: label_counts[label] for label in sorted(LABELS)},
                          "factual_conflict_detected": label_counts["contradiction"] > 0,
                          "source_contamination_detected": final_state == "source_contamination_detected",
                          "safety_label": safety, "final_state": final_state,
                          "claims": claim_rows, "source_role_used_for_decision": False,
                          "gold_or_target_labels_used": False,
                          "synthetic_only": answer.get("synthetic_only", False),
                          "requires_manual_review": True})
        graphs.append({"answer_id": aid, "question_id": answer["question_id"],
                       "nodes": nodes, "edges": edges,
                       "source_role_used_for_decision": False})
        if not claim_rows:
            review.append({"answer_id": aid, "claim_id": None, "reasons": ["no_factual_claims"]})
    summary = {"schema_version": SCHEMA, "answers": len(judgments), "claims": len(claims),
               "agent2_label_counts": dict(Counter(row["label"] for row in checks.values())),
               "final_state_counts": dict(Counter(row["final_state"] for row in judgments)),
               "answers_with_factual_conflict": sum(row["factual_conflict_detected"] for row in judgments),
               "answers_with_source_dependence_evidence": sum(row["source_contamination_detected"]
                                                              for row in judgments),
               "agent2a_attributions": len(attrs), "agent2b_eligible_claims": a2b_summary["eligible_claims"],
               "agent2b_complete_claims": len(effects), "review_queue_items": len(review),
               "graph_nodes": sum(len(row["nodes"]) for row in graphs),
               "graph_edges": sum(len(row["edges"]) for row in graphs),
               "model_called": False, "source_role_used_for_decision": False,
               "gold_or_target_labels_used": False,
               "performance_metrics_computed": False, "requires_manual_review": True,
               "input_sha256": hashes}
    write_jsonl(args.output_dir / "agent3_judgments.jsonl", judgments)
    write_jsonl(args.output_dir / "agent3_evidence_graphs.jsonl", graphs)
    write_jsonl(args.output_dir / "agent3_review_queue.jsonl", review)
    write_json(args.output_dir / "summary.json", summary)
    write_json(manifest_path, {"schema_version": SCHEMA, "input_sha256": hashes,
                               "model_called": False, "gold_or_target_labels_used": False})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag", type=Path, default=RAG)
    parser.add_argument("--agent1-results", type=Path, default=AGENT1_RESULTS)
    parser.add_argument("--agent1-claims", type=Path, default=AGENT1_CLAIMS)
    parser.add_argument("--agent2", type=Path, default=AGENT2)
    parser.add_argument("--agent2-summary", type=Path, default=AGENT2_SUMMARY)
    parser.add_argument("--agent2a", type=Path, default=AGENT2A)
    parser.add_argument("--agent2a-summary", type=Path, default=AGENT2A_SUMMARY)
    parser.add_argument("--agent2b-claims", type=Path, default=AGENT2B_CLAIMS)
    parser.add_argument("--agent2b", type=Path, default=AGENT2B)
    parser.add_argument("--agent2b-summary", type=Path, default=AGENT2B_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
