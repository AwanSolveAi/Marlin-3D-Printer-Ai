"""Retrieval-only evaluation for the existing Marlin 43-question benchmark.

This module intentionally performs no answer generation and makes no Ollama
requests. Relevance is judged from the benchmark's expected terms and answer
keywords against retrieved chunk content and metadata.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from answer_generator import MarlinAnswerAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_questions.json"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_evaluation_results.json"
)
SEARCH_DEPTH = 80

SEARCH_FIELDS = (
    "document",
    "category",
    "document_type",
    "source",
    "section",
    "gcode",
    "configuration",
    "content",
)


def _compact(text: Any) -> str:
    """Normalize presentation variants such as `VS Code` and `VSCode`."""

    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def term_present(term: str, text: str) -> bool:
    term = str(term or "").strip()
    text = str(text or "")

    if re.fullmatch(r"[GMT]\d+", term, flags=re.IGNORECASE) or "_" in term:
        return bool(
            re.search(
                rf"(?<![A-Z0-9_]){re.escape(term)}(?![A-Z0-9_])",
                text,
                flags=re.IGNORECASE,
            )
        )

    return _compact(term) in _compact(text)


def result_text(result: dict[str, Any]) -> str:
    return " ".join(str(result.get(field, "")) for field in SEARCH_FIELDS)


def is_relevant(result: dict[str, Any], item: dict[str, Any]) -> bool:
    """Require every expected term and at least one evidence keyword."""

    text = result_text(result)
    expected_terms = item.get("expected_terms", [])
    keywords = item.get("expected_answer_keywords", [])
    terms_match = all(term_present(term, text) for term in expected_terms)
    keyword_match = not keywords or any(term_present(word, text) for word in keywords)
    return terms_match and keyword_match


def expected_route(item: dict[str, Any]) -> str:
    if not item.get("should_answer", True):
        return "unsupported"
    if item.get("category") == "features":
        return "feature"
    return str(item.get("category", "general"))


def expected_evidence(item: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "all_expected_terms": list(item.get("expected_terms", [])),
        "any_evidence_keyword": list(item.get("expected_answer_keywords", [])),
    }


def evaluate() -> dict[str, Any]:
    questions = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    agent = MarlinAnswerAgent()
    cases: list[dict[str, Any]] = []
    retrieval_latencies: list[float] = []

    for item in questions:
        question = str(item["question"])
        should_answer = bool(item.get("should_answer", True))
        rejected = agent.looks_unrelated(question)
        route = "unsupported" if rejected else agent.detect_query_type(question)
        results: list[dict[str, Any]] = []
        latency_ms: float | None = None

        # Run supported cases and unsupported cases missed by the rejection gate.
        if should_answer or not rejected:
            expanded = agent.expand_query(question)
            started = time.perf_counter()
            results = agent.retriever.retrieve(expanded, top_k=SEARCH_DEPTH)
            latency_ms = (time.perf_counter() - started) * 1000
            retrieval_latencies.append(latency_ms)

        ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if is_relevant(result, item)
        ]
        rank = ranks[0] if ranks else None
        top = results[0] if results else {}
        top_text = result_text(top) if top else ""
        expected_codes = [
            term
            for term in item.get("expected_terms", [])
            if re.fullmatch(r"[GMT]\d+", term, flags=re.IGNORECASE)
        ]
        expected_symbols = [
            term
            for term in item.get("expected_terms", [])
            if "_" in term or term == "MOTHERBOARD"
        ]

        cases.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "question": question,
                "should_answer": should_answer,
                "expected_evidence": expected_evidence(item),
                "expected_route": expected_route(item),
                "actual_route": route,
                "routing_correct": route == expected_route(item),
                "unsupported_rejected": rejected if not should_answer else None,
                "expected_evidence_rank": rank,
                "top_1_correct": rank == 1 if should_answer else None,
                "actual_top_1": {
                    "chunk_id": top.get("chunk_id"),
                    "document": top.get("document"),
                    "section": top.get("section"),
                }
                if top
                else None,
                "retrieval_latency_ms": round(latency_ms, 3)
                if latency_ms is not None
                else None,
                "exact_code_top_1_correct": all(
                    term_present(code, top_text) for code in expected_codes
                )
                if expected_codes
                else None,
                "config_symbol_top_1_correct": all(
                    term_present(symbol, top_text) for symbol in expected_symbols
                )
                if expected_symbols
                else None,
            }
        )

    supported = [case for case in cases if case["should_answer"]]
    unsupported = [case for case in cases if not case["should_answer"]]

    code_cases: list[bool] = []
    config_cases: list[bool] = []
    for case in supported:
        if case["exact_code_top_1_correct"] is not None:
            code_cases.append(case["exact_code_top_1_correct"])
        if case["config_symbol_top_1_correct"] is not None:
            config_cases.append(case["config_symbol_top_1_correct"])

    metrics = {
        "supported_questions": len(supported),
        "unsupported_questions": len(unsupported),
        "top_1_accuracy": sum(case["top_1_correct"] for case in supported)
        / len(supported),
        "recall_at_3": sum(
            bool(case["expected_evidence_rank"] and case["expected_evidence_rank"] <= 3)
            for case in supported
        )
        / len(supported),
        "recall_at_5": sum(
            bool(case["expected_evidence_rank"] and case["expected_evidence_rank"] <= 5)
            for case in supported
        )
        / len(supported),
        "mrr": sum(
            1 / case["expected_evidence_rank"]
            if case["expected_evidence_rank"]
            else 0
            for case in supported
        )
        / len(supported),
        "exact_code_top_1_accuracy": sum(code_cases) / len(code_cases),
        "exact_code_cases": len(code_cases),
        "config_symbol_top_1_accuracy": sum(config_cases) / len(config_cases),
        "config_symbol_cases": len(config_cases),
        "unsupported_rejection_accuracy": sum(
            case["unsupported_rejected"] for case in unsupported
        )
        / len(unsupported),
        "query_routing_accuracy": sum(case["routing_correct"] for case in cases)
        / len(cases),
        "average_retrieval_latency_ms": statistics.fmean(retrieval_latencies),
        "retrieval_calls": len(retrieval_latencies),
    }

    failures = [
        case
        for case in cases
        if (case["should_answer"] and not case["top_1_correct"])
        or (not case["should_answer"] and not case["unsupported_rejected"])
        or not case["routing_correct"]
    ]
    return {
        "method": {
            "relevance": "all expected_terms and at least one expected_answer_keyword in one chunk",
            "string_matching": "case-insensitive compact text; exact boundaries for G/M/T codes and underscore symbols",
            "ranking_scope": "37 supported questions",
            "routing_scope": "all 43 questions",
            "latency_scope": "retrieval calls only; model initialization excluded",
            "search_depth": SEARCH_DEPTH,
        },
        "metrics": metrics,
        "failures_and_ambiguities": failures,
        "cases": cases,
    }


def main() -> None:
    report = evaluate()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"Failures or ambiguities: {len(report['failures_and_ambiguities'])}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
