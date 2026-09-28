# ============================================================
# MARLIN AI SUPPORT AGENT
# Automated RAG Evaluation
# Version: Evaluation v2
#
# Purpose:
#   Evaluate the complete Marlin RAG pipeline:
#
#       Question
#           ↓
#       Hybrid Retrieval
#           ↓
#       Evidence Gate
#           ↓
#       Context Builder
#           ↓
#       Ollama / llama3.2:3b
#           ↓
#       Output Validation
#
# IMPORTANT:
#   This evaluator is matched specifically to:
#
#       MarlinAnswerAgent
#       agent.generate(query, show_sources=False)
#
#   It does NOT modify:
#       - retrieve.py
#       - context_builder.py
#       - answer_generator.py
#       - index
#       - chunks
# ============================================================

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation_questions.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_results.json"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_report.txt"
)


# ============================================================
# IMPORT APPLICATION
# ============================================================

from answer_generator import MarlinAnswerAgent


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_EXPECTED_TERMS = []

REFUSAL_PHRASES = [
    "does not contain enough information",
    "doesn't contain enough information",
    "not enough information",
    "cannot answer this confidently",
    "can't answer this confidently",
    "insufficient information",
    "not enough evidence",
    "documentation does not provide enough",
    "documentation doesn't provide enough",
    "not documented in the provided",
    "not covered by the provided documentation",
    "cannot be answered from the documentation",
    "can't be answered from the documentation",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: Any) -> str:
    """
    Normalize text for robust evaluation.
    """

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def term_present(
    term: str,
    text: str,
) -> bool:
    """
    Check whether a technical term appears in text.

    Uses a case-insensitive substring check because Marlin
    identifiers may contain underscores, dots, or symbols.
    """

    term_normalized = normalize_text(term)
    text_normalized = normalize_text(text)

    if not term_normalized:
        return False

    return term_normalized in text_normalized


def keyword_present(
    keyword: str,
    text: str,
) -> bool:
    """
    Case-insensitive keyword matching.
    """

    return term_present(
        keyword,
        text,
    )


def is_refusal(
    answer: str,
) -> bool:
    """
    Determine whether the answer is an application refusal.
    """

    normalized = normalize_text(
        answer
    )

    if not normalized:
        return True

    for phrase in REFUSAL_PHRASES:

        if phrase in normalized:

            return True

    return False


def extract_gcodes(
    text: str,
) -> List[str]:
    """
    Extract G-code and M-code identifiers.
    """

    matches = re.findall(
        r"\b[GM]\d+(?:\.\d+)?\b",
        str(text or "").upper(),
    )

    return list(
        dict.fromkeys(matches)
    )


# ============================================================
# EVALUATION FILE
# ============================================================

def load_evaluation_questions() -> List[Dict[str, Any]]:
    """
    Load evaluation questions from JSON.
    """

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            "Evaluation file not found:\n"
            f"{EVALUATION_FILE}"
        )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # Support either:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or:
    #
    # {
    #   "questions": [...]
    # }
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        questions = data

    elif isinstance(
        data,
        dict,
    ):

        questions = (
            data.get("questions")
            or data.get("evaluation_questions")
            or []
        )

    else:

        raise ValueError(
            "Evaluation JSON must contain a list "
            "or an object containing 'questions'."
        )

    if not questions:

        raise ValueError(
            "No evaluation questions found."
        )

    return questions


# ============================================================
# FIELD HELPERS
# ============================================================

def get_question_id(
    item: Dict[str, Any],
    index: int,
) -> str:

    return str(
        item.get(
            "id",
            f"test_{index:03d}",
        )
    )


def get_category(
    item: Dict[str, Any],
) -> str:

    return str(
        item.get(
            "category",
            "uncategorized",
        )
    )


def get_question(
    item: Dict[str, Any],
) -> str:

    return str(
        item.get(
            "question",
            "",
        )
    ).strip()


def get_expected_terms(
    item: Dict[str, Any],
) -> List[str]:

    value = item.get(
        "expected_terms",
        DEFAULT_EXPECTED_TERMS,
    )

    if value is None:

        return []

    if isinstance(
        value,
        str,
    ):

        return [value]

    if isinstance(
        value,
        list,
    ):

        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    return []


def get_expected_keywords(
    item: Dict[str, Any],
) -> List[str]:

    value = item.get(
        "expected_answer_keywords",
        [],
    )

    if value is None:

        return []

    if isinstance(
        value,
        str,
    ):

        return [value]

    if isinstance(
        value,
        list,
    ):

        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    return []


def get_should_answer(
    item: Dict[str, Any],
) -> bool:

    value = item.get(
        "should_answer",
        True,
    )

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        str,
    ):

        return value.strip().lower() in {
            "true",
            "yes",
            "1",
            "supported",
            "answer",
        }

    return bool(value)


# ============================================================
# EXPECTED TERM EVALUATION
# ============================================================

def evaluate_expected_terms(
    answer: str,
    expected_terms: List[str],
) -> Dict[str, Any]:
    """
    Evaluate technical identifier coverage.
    """

    if not expected_terms:

        return {
            "total": 0,
            "matched": 0,
            "coverage": 1.0,
            "matched_terms": [],
            "missing_terms": [],
        }

    matched = []
    missing = []

    for term in expected_terms:

        if term_present(
            term,
            answer,
        ):

            matched.append(
                term
            )

        else:

            missing.append(
                term
            )

    coverage = (
        len(matched)
        / len(expected_terms)
    )

    return {
        "total": len(expected_terms),
        "matched": len(matched),
        "coverage": round(
            coverage,
            4,
        ),
        "matched_terms": matched,
        "missing_terms": missing,
    }


# ============================================================
# EXPECTED KEYWORD EVALUATION
# ============================================================

def evaluate_expected_keywords(
    answer: str,
    expected_keywords: List[str],
) -> Dict[str, Any]:
    """
    Evaluate expected answer-content keywords.
    """

    if not expected_keywords:

        return {
            "total": 0,
            "matched": 0,
            "coverage": 1.0,
            "matched_keywords": [],
            "missing_keywords": [],
        }

    matched = []
    missing = []

    for keyword in expected_keywords:

        if keyword_present(
            keyword,
            answer,
        ):

            matched.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )

    coverage = (
        len(matched)
        / len(expected_keywords)
    )

    return {
        "total": len(expected_keywords),
        "matched": len(matched),
        "coverage": round(
            coverage,
            4,
        ),
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


# ============================================================
# CODE SAFETY EVALUATION
# ============================================================

def evaluate_answer_codes(
    answer: str,
    sources: Any,
) -> Dict[str, Any]:
    """
    Check whether generated G/M codes are supported by the
    retrieved source content.

    This is an additional evaluator-side hallucination signal.
    """

    answer_codes = set(
        extract_gcodes(
            answer
        )
    )

    source_text_parts = []

    if isinstance(
        sources,
        list,
    ):

        for source in sources:

            if isinstance(
                source,
                dict,
            ):

                source_text_parts.append(
                    str(
                        source.get(
                            "content",
                            "",
                        )
                    )
                )

                source_text_parts.append(
                    str(
                        source.get(
                            "text",
                            "",
                        )
                    )
                )

                source_text_parts.append(
                    str(
                        source.get(
                            "source",
                            "",
                        )
                    )
                )

                source_text_parts.append(
                    str(
                        source.get(
                            "document",
                            "",
                        )
                    )
                )

            else:

                source_text_parts.append(
                    str(source)
                )

    source_text = " ".join(
        source_text_parts
    )

    source_codes = set(
        extract_gcodes(
            source_text
        )
    )

    unsupported_codes = sorted(
        answer_codes
        - source_codes
    )

    return {
        "answer_codes": sorted(
            answer_codes
        ),
        "source_codes": sorted(
            source_codes
        ),
        "unsupported_codes": unsupported_codes,
        "safe": not unsupported_codes,
    }


# ============================================================
# SINGLE TEST
# ============================================================

def evaluate_question(
    agent: MarlinAnswerAgent,
    item: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Evaluate one question through the real application
    pipeline.
    """

    question_id = get_question_id(
        item,
        index,
    )

    category = get_category(
        item
    )

    question = get_question(
        item
    )

    expected_terms = (
        get_expected_terms(
            item
        )
    )

    expected_keywords = (
        get_expected_keywords(
            item
        )
    )

    should_answer = (
        get_should_answer(
            item
        )
    )

    print()
    print(
        "-" * 90
    )

    print(
        f"[{index}] {question_id}"
    )

    print(
        f"Category : {category}"
    )

    print(
        f"Question : {question}"
    )

    start_time = time.perf_counter()

    try:

        result = agent.generate(
            question,
            show_sources=False,
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        answer = str(
            result.get(
                "answer",
                "",
            )
        ).strip()

        grounded = bool(
            result.get(
                "grounded",
                False,
            )
        )

        valid = bool(
            result.get(
                "valid",
                False,
            )
        )

        validation_reason = str(
            result.get(
                "validation_reason",
                "",
            )
        )

        sources = result.get(
            "sources",
            [],
        )

        # ----------------------------------------------------
        # Refusal
        # ----------------------------------------------------

        refused = is_refusal(
            answer
        )

        # ----------------------------------------------------
        # Term coverage
        # ----------------------------------------------------

        term_result = (
            evaluate_expected_terms(
                answer,
                expected_terms,
            )
        )

        # ----------------------------------------------------
        # Keyword coverage
        # ----------------------------------------------------

        keyword_result = (
            evaluate_expected_keywords(
                answer,
                expected_keywords,
            )
        )

        # ----------------------------------------------------
        # Code safety
        # ----------------------------------------------------

        code_result = (
            evaluate_answer_codes(
                answer,
                sources,
            )
        )

        # ----------------------------------------------------
        # Determine pass/fail
        # ----------------------------------------------------

        if should_answer:

            passed = (
                valid
                and grounded
                and not refused
                and term_result[
                    "coverage"
                ] >= 0.5
                and keyword_result[
                    "coverage"
                ] >= 0.5
                and code_result[
                    "safe"
                ]
            )

            failure_reasons = []

            if not valid:
                failure_reasons.append(
                    "invalid_answer"
                )

            if not grounded:
                failure_reasons.append(
                    "not_grounded"
                )

            if refused:
                failure_reasons.append(
                    "unexpected_refusal"
                )

            if (
                term_result["coverage"]
                < 0.5
            ):
                failure_reasons.append(
                    "low_expected_term_coverage"
                )

            if (
                keyword_result["coverage"]
                < 0.5
            ):
                failure_reasons.append(
                    "low_keyword_coverage"
                )

            if not code_result["safe"]:
                failure_reasons.append(
                    "unsupported_gcode_or_mcode"
                )

        else:

            passed = (
                refused
                or not grounded
            )

            failure_reasons = []

            if not passed:

                failure_reasons.append(
                    "unsupported_question_was_answered"
                )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print(
            f"Result   : "
            f"{'PASS' if passed else 'FAIL'}"
        )

        print(
            f"Grounded : {grounded}"
        )

        print(
            f"Valid    : {valid}"
        )

        print(
            f"Refused  : {refused}"
        )

        print(
            f"Term Cov : "
            f"{term_result['coverage']:.2%}"
        )

        print(
            f"Keyword  : "
            f"{keyword_result['coverage']:.2%}"
        )

        print(
            f"Code Safe: "
            f"{code_result['safe']}"
        )

        print(
            f"Latency  : "
            f"{latency:.2f}s"
        )

        print(
            "Answer:"
        )

        print(
            answer
        )

        if failure_reasons:

            print(
                "Failures:"
            )

            for reason in failure_reasons:

                print(
                    f"  - {reason}"
                )

        return {
            "id": question_id,
            "category": category,
            "question": question,
            "expected_terms": expected_terms,
            "expected_answer_keywords": (
                expected_keywords
            ),
            "should_answer": should_answer,
            "passed": passed,
            "failure_reasons": failure_reasons,
            "answer": answer,
            "grounded": grounded,
            "valid": valid,
            "refused": refused,
            "validation_reason": validation_reason,
            "expected_term_evaluation": term_result,
            "keyword_evaluation": keyword_result,
            "code_evaluation": code_result,
            "latency_seconds": round(
                latency,
                4,
            ),
        }

    except Exception as exc:

        latency = (
            time.perf_counter()
            - start_time
        )

        print(
            "Result   : ERROR"
        )

        print(
            f"Error    : {exc}"
        )

        return {
            "id": question_id,
            "category": category,
            "question": question,
            "expected_terms": expected_terms,
            "expected_answer_keywords": (
                expected_keywords
            ),
            "should_answer": should_answer,
            "passed": False,
            "failure_reasons": [
                "execution_error"
            ],
            "answer": "",
            "grounded": False,
            "valid": False,
            "refused": False,
            "validation_reason": "",
            "expected_term_evaluation": {
                "total": len(
                    expected_terms
                ),
                "matched": 0,
                "coverage": 0.0,
                "matched_terms": [],
                "missing_terms": expected_terms,
            },
            "keyword_evaluation": {
                "total": len(
                    expected_keywords
                ),
                "matched": 0,
                "coverage": 0.0,
                "matched_keywords": [],
                "missing_keywords": expected_keywords,
            },
            "code_evaluation": {
                "answer_codes": [],
                "source_codes": [],
                "unsupported_codes": [],
                "safe": False,
            },
            "latency_seconds": round(
                latency,
                4,
            ),
            "error": str(exc),
        }


# ============================================================
# CATEGORY SUMMARY
# ============================================================

def build_category_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build pass/fail statistics per category.
    """

    categories: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for result in results:

        category = result.get(
            "category",
            "uncategorized",
        )

        categories.setdefault(
            category,
            [],
        ).append(
            result
        )

    summary = {}

    for category, items in categories.items():

        total = len(items)

        passed = sum(
            1
            for item in items
            if item.get(
                "passed",
                False,
            )
        )

        latencies = [
            float(
                item.get(
                    "latency_seconds",
                    0.0,
                )
            )
            for item in items
        ]

        term_coverages = [
            float(
                item.get(
                    "expected_term_evaluation",
                    {},
                ).get(
                    "coverage",
                    0.0,
                )
            )
            for item in items
        ]

        keyword_coverages = [
            float(
                item.get(
                    "keyword_evaluation",
                    {},
                ).get(
                    "coverage",
                    0.0,
                )
            )
            for item in items
        ]

        summary[category] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(
                passed / total
                if total
                else 0.0,
                4,
            ),
            "average_latency_seconds": round(
                sum(latencies)
                / len(latencies)
                if latencies
                else 0.0,
                4,
            ),
            "average_term_coverage": round(
                sum(term_coverages)
                / len(term_coverages)
                if term_coverages
                else 0.0,
                4,
            ),
            "average_keyword_coverage": round(
                sum(keyword_coverages)
                / len(keyword_coverages)
                if keyword_coverages
                else 0.0,
                4,
            ),
        }

    return summary


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def build_global_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build overall evaluation metrics.
    """

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result.get(
            "passed",
            False,
        )
    )

    supported_questions = [
        result
        for result in results
        if result.get(
            "should_answer",
            True,
        )
    ]

    unsupported_questions = [
        result
        for result in results
        if not result.get(
            "should_answer",
            True,
        )
    ]

    supported_passed = sum(
        1
        for result in supported_questions
        if result.get(
            "passed",
            False,
        )
    )

    unsupported_passed = sum(
        1
        for result in unsupported_questions
        if result.get(
            "passed",
            False,
        )
    )

    latencies = [
        float(
            result.get(
                "latency_seconds",
                0.0,
            )
        )
        for result in results
    ]

    term_coverages = [
        float(
            result.get(
                "expected_term_evaluation",
                {},
            ).get(
                "coverage",
                0.0,
            )
        )
        for result in supported_questions
    ]

    keyword_coverages = [
        float(
            result.get(
                "keyword_evaluation",
                {},
            ).get(
                "coverage",
                0.0,
            )
        )
        for result in supported_questions
    ]

    unexpected_refusals = sum(
        1
        for result in supported_questions
        if result.get(
            "refused",
            False,
        )
    )

    unsupported_answered = sum(
        1
        for result in unsupported_questions
        if not result.get(
            "refused",
            False,
        )
    )

    unsupported_code_cases = sum(
        1
        for result in results
        if not result.get(
            "code_evaluation",
            {},
        ).get(
            "safe",
            False,
        )
    )

    return {
        "total_questions": total,
        "passed_questions": passed,
        "failed_questions": (
            total - passed
        ),
        "overall_pass_rate": round(
            passed / total
            if total
            else 0.0,
            4,
        ),
        "supported_questions": len(
            supported_questions
        ),
        "supported_passed": supported_passed,
        "supported_pass_rate": round(
            supported_passed
            / len(supported_questions)
            if supported_questions
            else 0.0,
            4,
        ),
        "unsupported_questions": len(
            unsupported_questions
        ),
        "unsupported_passed": unsupported_passed,
        "unsupported_detection_rate": round(
            unsupported_passed
            / len(unsupported_questions)
            if unsupported_questions
            else 0.0,
            4,
        ),
        "unexpected_refusals": (
            unexpected_refusals
        ),
        "unsupported_questions_answered": (
            unsupported_answered
        ),
        "unsupported_code_cases": (
            unsupported_code_cases
        ),
        "average_latency_seconds": round(
            sum(latencies)
            / len(latencies)
            if latencies
            else 0.0,
            4,
        ),
        "min_latency_seconds": round(
            min(latencies)
            if latencies
            else 0.0,
            4,
        ),
        "max_latency_seconds": round(
            max(latencies)
            if latencies
            else 0.0,
            4,
        ),
        "average_supported_term_coverage": round(
            sum(term_coverages)
            / len(term_coverages)
            if term_coverages
            else 0.0,
            4,
        ),
        "average_supported_keyword_coverage": round(
            sum(keyword_coverages)
            / len(keyword_coverages)
            if keyword_coverages
            else 0.0,
            4,
        ),
    }


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_text_report(
    results: List[Dict[str, Any]],
    global_summary: Dict[str, Any],
    category_summary: Dict[str, Dict[str, Any]],
) -> str:
    """
    Generate a human-readable evaluation report.
    """

    lines = []

    lines.append(
        "=" * 90
    )

    lines.append(
        "MARLIN AI SUPPORT AGENT"
    )

    lines.append(
        "AUTOMATED RAG EVALUATION REPORT"
    )

    lines.append(
        "=" * 90
    )

    lines.append("")

    lines.append(
        "GLOBAL SUMMARY"
    )

    lines.append(
        "-" * 90
    )

    lines.append(
        f"Total questions              : "
        f"{global_summary['total_questions']}"
    )

    lines.append(
        f"Passed questions             : "
        f"{global_summary['passed_questions']}"
    )

    lines.append(
        f"Failed questions             : "
        f"{global_summary['failed_questions']}"
    )

    lines.append(
        f"Overall pass rate            : "
        f"{global_summary['overall_pass_rate']:.2%}"
    )

    lines.append(
        f"Supported-question pass rate : "
        f"{global_summary['supported_pass_rate']:.2%}"
    )

    lines.append(
        f"Unsupported detection rate   : "
        f"{global_summary['unsupported_detection_rate']:.2%}"
    )

    lines.append(
        f"Unexpected refusals          : "
        f"{global_summary['unexpected_refusals']}"
    )

    lines.append(
        f"Unsupported questions answered: "
        f"{global_summary['unsupported_questions_answered']}"
    )

    lines.append(
        f"Unsupported code cases       : "
        f"{global_summary['unsupported_code_cases']}"
    )

    lines.append(
        f"Average latency              : "
        f"{global_summary['average_latency_seconds']:.2f}s"
    )

    lines.append(
        f"Minimum latency              : "
        f"{global_summary['min_latency_seconds']:.2f}s"
    )

    lines.append(
        f"Maximum latency              : "
        f"{global_summary['max_latency_seconds']:.2f}s"
    )

    lines.append(
        f"Average term coverage        : "
        f"{global_summary['average_supported_term_coverage']:.2%}"
    )

    lines.append(
        f"Average keyword coverage     : "
        f"{global_summary['average_supported_keyword_coverage']:.2%}"
    )

    lines.append("")

    lines.append(
        "CATEGORY SUMMARY"
    )

    lines.append(
        "-" * 90
    )

    for category in sorted(
        category_summary
    ):

        summary = category_summary[
            category
        ]

        lines.append(
            f"{category}"
        )

        lines.append(
            f"  Total       : "
            f"{summary['total']}"
        )

        lines.append(
            f"  Passed      : "
            f"{summary['passed']}"
        )

        lines.append(
            f"  Failed      : "
            f"{summary['failed']}"
        )

        lines.append(
            f"  Pass rate   : "
            f"{summary['pass_rate']:.2%}"
        )

        lines.append(
            f"  Avg latency : "
            f"{summary['average_latency_seconds']:.2f}s"
        )

        lines.append(
            f"  Term cover  : "
            f"{summary['average_term_coverage']:.2%}"
        )

        lines.append(
            f"  Keyword cov : "
            f"{summary['average_keyword_coverage']:.2%}"
        )

        lines.append("")

    # --------------------------------------------------------
    # Failed tests
    # --------------------------------------------------------

    failed = [
        result
        for result in results
        if not result.get(
            "passed",
            False,
        )
    ]

    lines.append(
        "FAILED TESTS"
    )

    lines.append(
        "-" * 90
    )

    if not failed:

        lines.append(
            "None. All evaluation tests passed."
        )

    else:

        for result in failed:

            lines.append(
                f"[{result.get('id')}] "
                f"{result.get('question')}"
            )

            reasons = result.get(
                "failure_reasons",
                [],
            )

            if reasons:

                lines.append(
                    "  Reasons: "
                    + ", ".join(
                        reasons
                    )
                )

            lines.append(
                "  Answer: "
                + str(
                    result.get(
                        "answer",
                        "",
                    )
                )
            )

            lines.append("")

    # --------------------------------------------------------
    # Successful tests
    # --------------------------------------------------------

    lines.append(
        "ALL TEST RESULTS"
    )

    lines.append(
        "-" * 90
    )

    for result in results:

        status = (
            "PASS"
            if result.get(
                "passed",
                False,
            )
            else "FAIL"
        )

        lines.append(
            f"{status:4} | "
            f"{result.get('id')} | "
            f"{result.get('category')} | "
            f"{result.get('latency_seconds', 0):.2f}s"
        )

    lines.append("")

    lines.append(
        "=" * 90
    )

    lines.append(
        "END OF REPORT"
    )

    lines.append(
        "=" * 90
    )

    return "\n".join(
        lines
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: List[Dict[str, Any]],
    global_summary: Dict[str, Any],
    category_summary: Dict[str, Dict[str, Any]],
) -> None:
    """
    Save machine-readable and human-readable reports.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "evaluation_file": str(
            EVALUATION_FILE
        ),
        "results": results,
        "global_summary": global_summary,
        "category_summary": category_summary,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    report = generate_text_report(
        results,
        global_summary,
        category_summary,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 90
    )

    print(
        "MARLIN AI SUPPORT AGENT"
    )

    print(
        "AUTOMATED RAG EVALUATION"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    questions = (
        load_evaluation_questions()
    )

    print()
    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print()

    # --------------------------------------------------------
    # Initialize exact application class
    # --------------------------------------------------------

    print(
        "Initializing MarlinAnswerAgent..."
    )

    agent = (
        MarlinAnswerAgent()
    )

    print()
    print(
        "Answer generator: OK"
    )

    print()

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    results = []

    evaluation_start = (
        time.perf_counter()
    )

    for index, item in enumerate(
        questions,
        start=1,
    ):

        result = evaluate_question(
            agent,
            item,
            index,
        )

        results.append(
            result
        )

    total_evaluation_time = (
        time.perf_counter()
        - evaluation_start
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    global_summary = (
        build_global_summary(
            results
        )
    )

    category_summary = (
        build_category_summary(
            results
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        global_summary,
        category_summary,
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 90
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Total tests       : "
        f"{global_summary['total_questions']}"
    )

    print(
        f"Passed            : "
        f"{global_summary['passed_questions']}"
    )

    print(
        f"Failed            : "
        f"{global_summary['failed_questions']}"
    )

    print(
        f"Overall pass rate : "
        f"{global_summary['overall_pass_rate']:.2%}"
    )

    print(
        f"Supported pass    : "
        f"{global_summary['supported_pass_rate']:.2%}"
    )

    print(
        f"Unsupported detect: "
        f"{global_summary['unsupported_detection_rate']:.2%}"
    )

    print(
        f"Unexpected refusal: "
        f"{global_summary['unexpected_refusals']}"
    )

    print(
        f"Avg latency       : "
        f"{global_summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"Total runtime     : "
        f"{total_evaluation_time:.2f}s"
    )

    print()

    print(
        f"JSON results:"
    )

    print(
        RESULTS_FILE
    )

    print()

    print(
        f"Text report:"
    )

    print(
        REPORT_FILE
    )

    print()
    print(
        "=" * 90
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()