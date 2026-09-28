"""
Marlin AI Support Agent
Context Builder

Builds a clean, structured context package from retrieved
Marlin documentation chunks.

This module does NOT generate the final answer.
It prepares evidence for the LLM/agent layer.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_MAX_CHUNKS = 6
DEFAULT_MAX_CHARS_PER_CHUNK = 5000
DEFAULT_MAX_CONTEXT_CHARS = 24000


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_content(text: str) -> str:
    """
    Clean retrieved content while preserving technical formatting.
    """

    if not text:
        return ""

    text = str(text)

    # Remove accidental null characters.
    text = text.replace("\x00", " ")

    # Normalize Windows line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive blank lines.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


# ============================================================
# FORMAT ONE RESULT
# ============================================================

def format_result(
    result: dict[str, Any],
    rank: int,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> str:
    """
    Convert one retrieval result into a structured evidence block.
    """

    document = str(
        result.get("document", "")
    ).strip()

    category = str(
        result.get("category", "")
    ).strip()

    document_type = str(
        result.get("document_type", "")
    ).strip()

    source = str(
        result.get("source", "")
    ).strip()

    section = str(
        result.get("section", "")
    ).strip()

    gcode = str(
        result.get("gcode", "")
    ).strip()

    configuration = str(
        result.get("configuration", "")
    ).strip()

    content = clean_content(
        result.get("content", "")
    )

    if len(content) > max_chars:
        content = content[:max_chars] + "\n[Content truncated]"

    lines = [
        f"--- EVIDENCE {rank} ---",
        f"Document: {document or 'Unknown'}",
        f"Category: {category or 'Unknown'}",
        f"Document Type: {document_type or 'Unknown'}",
    ]

    if source:
        lines.append(f"Source: {source}")

    if section:
        lines.append(f"Section: {section}")

    if gcode:
        lines.append(f"G-code: {gcode}")

    if configuration:
        lines.append(
            f"Configuration: {configuration}"
        )

    lines.extend(
        [
            "",
            "Content:",
            content,
            "",
            f"--- END EVIDENCE {rank} ---",
        ]
    )

    return "\n".join(lines)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    query: str,
    results: list[dict[str, Any]],
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    """
    Build structured context from retrieval results.

    Returns:

        {
            "query": ...,
            "context": ...,
            "evidence": [...],
            "sources": [...],
            "num_results": ...,
        }
    """

    query = str(query).strip()

    if not query:
        return {
            "query": "",
            "context": "",
            "evidence": [],
            "sources": [],
            "num_results": 0,
        }

    if not results:
        return {
            "query": query,
            "context": "",
            "evidence": [],
            "sources": [],
            "num_results": 0,
        }

    selected_results = results[:max_chunks]

    evidence_blocks: list[str] = []
    evidence_data: list[dict[str, Any]] = []
    sources: list[str] = []

    total_chars = 0

    for rank, result in enumerate(
        selected_results,
        start=1,
    ):

        block = format_result(
            result,
            rank=rank,
            max_chars=max_chars_per_chunk,
        )

        # Respect total context limit.
        remaining = max_context_chars - total_chars

        if remaining <= 0:
            break

        if len(block) > remaining:

            block = (
                block[:remaining]
                + "\n[Context limit reached]"
            )

        evidence_blocks.append(block)

        evidence_data.append(
            {
                "rank": rank,
                "chunk_id": result.get(
                    "chunk_id",
                    "",
                ),
                "document": result.get(
                    "document",
                    "",
                ),
                "category": result.get(
                    "category",
                    "",
                ),
                "document_type": result.get(
                    "document_type",
                    "",
                ),
                "source": result.get(
                    "source",
                    "",
                ),
                "section": result.get(
                    "section",
                    "",
                ),
                "gcode": result.get(
                    "gcode",
                    "",
                ),
                "configuration": result.get(
                    "configuration",
                    "",
                ),
                "final_score": result.get(
                    "final_score",
                    0.0,
                ),
                "rrf_score": result.get(
                    "rrf_score",
                    0.0,
                ),
                "bm25_score": result.get(
                    "bm25_score",
                    0.0,
                ),
                "vector_score": result.get(
                    "vector_score",
                    0.0,
                ),
                "boost": result.get(
                    "boost",
                    0.0,
                ),
                "match_reasons": result.get(
                    "match_reasons",
                    [],
                ),
                "content": result.get(
                    "content",
                    "",
                ),
            }
        )

        source = str(
            result.get(
                "source",
                "",
            )
        ).strip()

        document = str(
            result.get(
                "document",
                "",
            )
        ).strip()

        source_name = source or document

        if source_name and source_name not in sources:
            sources.append(source_name)

        total_chars += len(block)

    context = "\n\n".join(
        evidence_blocks
    )

    return {
        "query": query,
        "context": context,
        "evidence": evidence_data,
        "sources": sources,
        "num_results": len(evidence_data),
    }


# ============================================================
# SIMPLE CONTEXT STRING
# ============================================================

def build_context_string(
    query: str,
    results: list[dict[str, Any]],
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> str:
    """
    Convenience function when only the final context string
    is required.
    """

    context_data = build_context(
        query=query,
        results=results,
        max_chunks=max_chunks,
    )

    return context_data["context"]


# ============================================================
# DEBUG PRINT
# ============================================================

def print_context(
    context_data: dict[str, Any],
) -> None:
    """
    Print the generated context package.
    """

    print("\n" + "=" * 80)
    print("MARLIN CONTEXT")
    print("=" * 80)

    print(
        f"\nQuery: {context_data.get('query', '')}"
    )

    print(
        f"Evidence chunks: "
        f"{context_data.get('num_results', 0)}"
    )

    print("\n" + "-" * 80)

    context = context_data.get(
        "context",
        "",
    )

    if context:
        print(context)
    else:
        print("No evidence available.")

    print("\n" + "=" * 80)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("MARLIN CONTEXT BUILDER")
    print("=" * 80)

    print(
        "\nThis module is designed to be imported by "
        "the Marlin AI Support Agent."
    )

    print(
        "\nExample:"
    )

    print(
        """
from context_builder import build_context

results = retriever.retrieve(
    query,
    top_k=8,
)

context_data = build_context(
    query,
    results,
)

print(context_data["context"])
"""
    )