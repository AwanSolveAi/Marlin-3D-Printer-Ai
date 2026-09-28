"""
Marlin AI Support Agent
Hybrid Retrieval Index Builder

Builds:
    - FAISS vector index
    - BM25 index
    - Metadata JSON

Input:
    data/processed/marlin_chunks.jsonl

Output:
    data/vector_store/marlin_faiss.index
    data/vector_store/marlin_metadata.json
    data/vector_store/marlin_bm25.pkl
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "marlin_chunks.jsonl"

VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

FAISS_FILE = VECTOR_DIR / "marlin_faiss.index"
METADATA_FILE = VECTOR_DIR / "marlin_metadata.json"
BM25_FILE = VECTOR_DIR / "marlin_bm25.pkl"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# SETTINGS
# ============================================================

MIN_TEXT_LENGTH = 50


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving useful Markdown/code content.
    """
    if not text:
        return ""

    text = str(text).replace("\x00", " ")

    # Normalize excessive whitespace but preserve line structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def tokenize_bm25(text: str) -> list[str]:
    """
    Tokenization designed for technical Marlin documentation.

    Important:
        M104, G29, M420, Configuration symbols, etc.
        must remain searchable as exact tokens.
    """

    text = text.lower()

    # Keep technical identifiers together.
    tokens = re.findall(
        r"""
        [a-z_][a-z0-9_]*
        |
        [gmst]\d+
        |
        \d+(?:\.\d+)?
        |
        [a-z]+\d+[a-z0-9_]*
        """,
        text,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return tokens


def extract_gcode_codes(text: str) -> list[str]:
    """
    Extract G/M/T/S-style commands such as:

        M104
        M109
        G28
        G29
        T0

    Also handles commands appearing inside Markdown code.
    """

    if not text:
        return []

    matches = re.findall(
        r"\b([GMT]\d+(?:\.\d+)?)\b",
        text,
        flags=re.IGNORECASE,
    )

    return sorted(set(x.upper() for x in matches))


def extract_configuration_symbols(text: str) -> list[str]:
    """
    Extract likely Marlin configuration constants.

    Examples:
        AUTO_BED_LEVELING_UBL
        GRID_MAX_POINTS
        MESH_MIN
        PROBING_MARGIN
        MOTHERBOARD
        SERIAL_PORT
    """

    if not text:
        return []

    candidates = re.findall(
        r"\b[A-Z][A-Z0-9_]{3,}\b",
        text,
    )

    excluded = {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "THIS",
        "FROM",
        "THAT",
        "MARLIN",
        "GCODE",
        "LCD",
        "EEPROM",
        "USB",
        "IDE",
        "CPU",
        "RAM",
        "SD",
    }

    return sorted(
        set(
            item
            for item in candidates
            if item not in excluded
        )
    )


def build_search_text(record: dict[str, Any]) -> str:
    """
    Create enriched searchable text.

    Metadata is deliberately included because queries such as:

        "M104"
        "configuration MOTHERBOARD"
        "UBL M420"

    should retrieve the correct technical material.
    """

    parts = [
        record.get("document", ""),
        record.get("category", ""),
        record.get("document_type", ""),
        record.get("source", ""),
        record.get("section", ""),
        record.get("gcode", ""),
        record.get("configuration", ""),
        record.get("content", ""),
    ]

    return "\n".join(
        str(part)
        for part in parts
        if part
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[dict[str, Any]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    records: list[dict[str, Any]] = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Invalid JSON at line {line_number}: {exc}"
                )
                continue

            content = clean_text(
                record.get("content", "")
            )

            if len(content) < MIN_TEXT_LENGTH:
                continue

            record["content"] = content

            record["gcode_codes"] = extract_gcode_codes(
                build_search_text(record)
            )

            record["configuration_symbols"] = (
                extract_configuration_symbols(
                    build_search_text(record)
                )
            )

            record["_search_text"] = build_search_text(record)

            records.append(record)

    return records


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(records: list[dict[str, Any]]):

    print("\nBuilding BM25 index...")

    tokenized_corpus = [
        tokenize_bm25(record["_search_text"])
        for record in records
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


# ============================================================
# BUILD FAISS
# ============================================================

def build_faiss(records: list[dict[str, Any]]):

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        record["_search_text"]
        for record in records
    ]

    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Inner product over normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal}")

    return index


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(records: list[dict[str, Any]]):

    metadata = []

    for record in records:

        item = {
            "chunk_id": record.get("chunk_id", ""),
            "document": record.get("document", ""),
            "category": record.get("category", ""),
            "document_type": record.get("document_type", ""),
            "source": record.get("source", ""),
            "section": record.get("section", ""),
            "gcode": record.get("gcode", ""),
            "configuration": record.get("configuration", ""),
            "length": record.get("length", len(record.get("content", ""))),
            "content": record.get("content", ""),
            "gcode_codes": record.get("gcode_codes", []),
            "configuration_symbols": record.get(
                "configuration_symbols",
                [],
            ),
        }

        metadata.append(item)

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# SAVE BM25
# ============================================================

def save_bm25(
    bm25,
    tokenized_corpus,
):

    payload = {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
    }

    with BM25_FILE.open("wb") as f:
        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MARLIN AI SUPPORT AGENT - INDEX BUILDER")
    print("=" * 80)

    print(f"\nInput:")
    print(INPUT_FILE)

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_chunks()

    print(f"\nLoaded chunks: {len(records)}")

    if not records:
        raise RuntimeError(
            "No usable chunks were found."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    gcode_count = sum(
        bool(record.get("gcode_codes"))
        for record in records
    )

    config_count = sum(
        bool(record.get("configuration_symbols"))
        for record in records
    )

    print(f"G-code-aware chunks: {gcode_count}")
    print(f"Configuration-aware chunks: {config_count}")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25, tokenized_corpus = build_bm25(records)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = build_faiss(records)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving indexes...")

    faiss.write_index(
        index,
        str(FAISS_FILE),
    )

    save_metadata(records)

    save_bm25(
        bm25,
        tokenized_corpus,
    )

    # --------------------------------------------------------
    # Remove internal search text from anything persisted
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INDEX BUILD COMPLETE")
    print("=" * 80)

    print(f"Chunks indexed       : {len(records)}")
    print(f"FAISS vectors        : {index.ntotal}")
    print(f"Embedding dimension  : {index.d}")
    print(f"G-code-aware chunks  : {gcode_count}")
    print(f"Config-aware chunks  : {config_count}")

    print("\nSaved:")
    print(f"FAISS    : {FAISS_FILE}")
    print(f"Metadata : {METADATA_FILE}")
    print(f"BM25     : {BM25_FILE}")

    print("=" * 80)


if __name__ == "__main__":
    main()