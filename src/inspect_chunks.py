"""
Inspect Marlin RAG chunks.

Usage:

    python src/inspect_chunks.py

Optional:

    python src/inspect_chunks.py gcode
    python src/inspect_chunks.py configuration
    python src/inspect_chunks.py M104
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marlin_chunks.jsonl"
)


def load_chunks():

    chunks = []

    with CHUNK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                chunks.append(
                    json.loads(line)
                )

    return chunks


def matches(chunk, query):

    query = query.lower()

    searchable = " ".join(
        [
            str(chunk.get("title", "")),
            str(chunk.get("category", "")),
            str(chunk.get("document_type", "")),
            str(chunk.get("source_file", "")),
            str(chunk.get("heading_path", "")),
            str(chunk.get("content", "")),
            " ".join(
                chunk.get(
                    "gcode_commands",
                    [],
                )
            ),
            " ".join(
                chunk.get(
                    "configuration_terms",
                    [],
                )
            ),
        ]
    ).lower()

    return query in searchable


def print_chunk(chunk):

    print("=" * 80)

    print(
        f"Chunk ID       : {chunk.get('chunk_id')}"
    )

    print(
        f"Document       : {chunk.get('title')}"
    )

    print(
        f"Category       : {chunk.get('category')}"
    )

    print(
        f"Document type  : {chunk.get('document_type')}"
    )

    print(
        f"Source         : {chunk.get('source_file')}"
    )

    print(
        f"Section        : {chunk.get('heading_path')}"
    )

    print(
        f"G-code         : "
        f"{', '.join(chunk.get('gcode_commands', []))}"
    )

    print(
        f"Configuration  : "
        f"{', '.join(chunk.get('configuration_terms', []))}"
    )

    print(
        f"Length         : {chunk.get('content_length')}"
    )

    print()
    print("CONTENT")
    print("-" * 80)

    print(
        chunk.get(
            "content",
            "",
        )
    )

    print()


def main():

    if not CHUNK_FILE.exists():

        print(
            f"ERROR: Chunk file does not exist:\n{CHUNK_FILE}"
        )

        print()
        print(
            "Run first:"
        )
        print(
            "python src/build_chunks.py"
        )

        return

    chunks = load_chunks()

    print()
    print("=" * 80)
    print("MARLIN CHUNK INSPECTOR")
    print("=" * 80)

    print(
        f"Total chunks: {len(chunks)}"
    )

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else None
    )

    if query:

        results = [
            chunk
            for chunk in chunks
            if matches(chunk, query)
        ]

        print(
            f"Search: {query}"
        )

        print(
            f"Matches: {len(results)}"
        )

        print()

        for chunk in results[:10]:
            print_chunk(chunk)

    else:

        print()
        print(
            "Showing first 5 chunks."
        )

        print(
            "You can search using:"
        )

        print(
            "python src/inspect_chunks.py M104"
        )

        print(
            "python src/inspect_chunks.py configuration"
        )

        print()

        for chunk in chunks[:5]:
            print_chunk(chunk)


if __name__ == "__main__":
    main()