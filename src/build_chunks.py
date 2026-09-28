"""
Marlin Documentation - High Quality Chunk Builder V2

Purpose:
    Convert processed Marlin documents into retrieval-optimized chunks.

Input:
    data/processed/marlin_documents.jsonl

Output:
    data/processed/marlin_chunks.jsonl

Design goals:
    - Preserve Marlin technical terminology
    - Preserve G-code commands
    - Preserve configuration names
    - Preserve headings
    - Preserve code blocks
    - Preserve tables
    - Avoid tiny meaningless chunks
    - Avoid excessive chunk fragmentation
    - Add useful metadata for hybrid retrieval
    - Provide consistent metadata for retrieval and citations
    - Preserve backward compatibility with existing retriever/index
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marlin_documents.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marlin_chunks.jsonl"
)


# ============================================================
# CHUNK SETTINGS
# ============================================================

# Target size in characters.
TARGET_CHARS = 1800

# Hard maximum.
MAX_CHARS = 2600

# Minimum useful chunk size.
MIN_CHARS = 180

# Overlap used when an extremely long technical line
# must be split.
OVERLAP_CHARS = 250


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Fix common mojibake while avoiding aggressive rewriting.
    """

    replacements = {
        "â€¦": "…",
        "â€”": "—",
        "â€“": "–",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
        "Â": "",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¡": "á",
        "Ã³": "ó",
        "Ã¼": "ü",
        "Ã±": "ñ",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def remove_html_noise(text: str) -> str:
    """
    Remove HTML presentation noise while preserving
    useful textual content.
    """

    # Remove HTML comments.
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )

    # Convert common line-breaking HTML elements.
    text = re.sub(
        r"<\s*(br|hr)\s*/?\s*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Preserve image alt text where possible.
    text = re.sub(
        r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>',
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Remove empty image markdown.
    text = re.sub(
        r"!\[\]\([^)]+\)",
        "",
        text,
    )

    # Remove remaining HTML tags.
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    return text


def clean_markdown(text: str) -> str:
    """
    Clean Markdown without destroying technical structure.
    """

    text = normalize_unicode(text)
    text = remove_html_noise(text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize tabs.
    text = text.replace("\t", "    ")

    # Remove trailing whitespace.
    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{4,}",
        "\n\n",
        text,
    )

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        # Preserve fenced code block delimiters.
        if line.strip().startswith("```"):
            cleaned_lines.append(
                line.rstrip()
            )
            continue

        cleaned_lines.append(
            re.sub(
                r"[ ]{3,}",
                "  ",
                line,
            ).rstrip()
        )

    return "\n".join(cleaned_lines).strip()


# ============================================================
# METADATA EXTRACTION
# ============================================================

# Matches:
#   G0
#   G1
#   G29
#   M420
#   M500
#   T0
#
# Also allows a suffix such as:
#   G29_CUSTOM
#
# The base command is retained.
GCODE_PATTERN = re.compile(
    r"\b([GMT]\d{1,4})(?:[-_][A-Za-z0-9]+)?\b",
    flags=re.IGNORECASE,
)


# Likely Marlin configuration constants.
#
# Examples:
#   AUTO_BED_LEVELING_BILINEAR
#   Z_SAFE_HOMING
#   EEPROM_SETTINGS
CONFIG_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,})\b"
)


def extract_gcode_commands(
    title: str,
    source_file: str,
    content: str,
) -> List[str]:
    """
    Extract likely Marlin G/M/T commands.
    """

    combined = (
        f"{title}\n"
        f"{source_file}\n"
        f"{content}"
    )

    matches = GCODE_PATTERN.findall(
        combined
    )

    commands = sorted(
        {
            match.upper()
            for match in matches
            if match
            and match.upper()[0]
            in {"G", "M", "T"}
        }
    )

    return commands


def extract_configuration_terms(
    content: str,
) -> List[str]:
    """
    Extract likely Marlin configuration constants.

    Examples:
        AUTO_BED_LEVELING_BILINEAR
        Z_SAFE_HOMING
        EEPROM_SETTINGS
    """

    candidates = CONFIG_PATTERN.findall(
        content
    )

    useful = []

    ignored = {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "THIS",
        "MARLIN",
        "GCODE",
        "HTML",
        "URL",
        "NOTE",
        "WARNING",
        "IMPORTANT",
        "EXAMPLE",
        "DEFAULT",
    }

    for item in candidates:

        if item in ignored:
            continue

        # Marlin configuration symbols generally contain
        # underscores.
        if "_" not in item:
            continue

        if len(item) < 5:
            continue

        useful.append(item)

    return sorted(
        set(useful)
    )[:50]


def detect_document_type(
    title: str,
    category: str,
    source_file: str,
) -> str:
    """
    Determine a useful document type from existing metadata.
    """

    source_lower = (
        source_file or ""
    ).lower()

    title_lower = (
        title or ""
    ).lower()

    category_lower = (
        category or ""
    ).lower()

    if category_lower == "gcode":
        return "gcode"

    if "configuration" in source_lower:
        return "configuration"

    if "feature" in source_lower:
        return "feature"

    if "hardware" in source_lower:
        return "hardware"

    if "setting" in source_lower:
        return "setting"

    if "install" in title_lower:
        return "installation"

    if "troubleshoot" in title_lower:
        return "troubleshooting"

    return category or "general"


# ============================================================
# SECTION PARSING
# ============================================================

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def split_into_sections(
    text: str,
) -> List[Tuple[str, str]]:
    """
    Split a Markdown document into logical heading-based sections.

    Returns:
        [
            (heading_path, section_text),
            ...
        ]
    """

    lines = text.splitlines()

    sections: List[
        Tuple[str, str]
    ] = []

    heading_stack: List[str] = []

    current_lines: List[str] = []

    def flush_current():
        if not current_lines:
            return

        content = "\n".join(
            current_lines
        ).strip()

        if not content:
            return

        heading_path = (
            " > ".join(
                heading_stack
            )
        )

        sections.append(
            (
                heading_path,
                content,
            )
        )

    in_code_block = False

    for line in lines:

        # Code blocks must not be interpreted as headings.
        if line.strip().startswith("```"):

            in_code_block = (
                not in_code_block
            )

            current_lines.append(
                line
            )

            continue

        if not in_code_block:

            match = HEADING_PATTERN.match(
                line
            )

            if match:

                flush_current()
                current_lines.clear()

                level = len(
                    match.group(1)
                )

                heading = (
                    match.group(2)
                    .strip()
                )

                # Remove headings at the same or
                # deeper level.
                while (
                    len(heading_stack)
                    >= level
                ):
                    heading_stack.pop()

                heading_stack.append(
                    heading
                )

                # Keep the heading in the section content.
                current_lines.append(
                    line
                )

                continue

        current_lines.append(
            line
        )

    flush_current()

    return sections


# ============================================================
# CODE BLOCK HANDLING
# ============================================================

def find_code_blocks(
    text: str,
) -> List[str]:
    """
    Return fenced code blocks contained in a section.
    """

    return re.findall(
        r"```.*?```",
        text,
        flags=re.DOTALL,
    )


# ============================================================
# LARGE SECTION SPLITTING
# ============================================================

def split_large_text(
    text: str,
    target: int = TARGET_CHARS,
    maximum: int = MAX_CHARS,
) -> List[str]:
    """
    Split large sections while trying to respect paragraphs,
    lists, sentences, and technical boundaries.
    """

    if len(text) <= maximum:
        return [
            text.strip()
        ]

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks: List[str] = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= target:

            current = candidate

            continue

        if current:
            chunks.append(
                current.strip()
            )

        # Handle an individual oversized paragraph.
        if len(paragraph) > maximum:

            subparts = re.split(
                r"(?<=[.!?])\s+"
                r"|(?=\n[-*]\s+)",
                paragraph,
            )

            subcurrent = ""

            for part in subparts:

                part = part.strip()

                if not part:
                    continue

                candidate_sub = (
                    f"{subcurrent} {part}"
                    if subcurrent
                    else part
                )

                if len(candidate_sub) <= target:

                    subcurrent = (
                        candidate_sub
                    )

                    continue

                if subcurrent:
                    chunks.append(
                        subcurrent.strip()
                    )

                # Extremely long technical line.
                if len(part) > maximum:

                    step = (
                        maximum
                        - OVERLAP_CHARS
                    )

                    # Safety against invalid step.
                    step = max(
                        step,
                        1,
                    )

                    for start in range(
                        0,
                        len(part),
                        step,
                    ):

                        piece = part[
                            start:
                            start + maximum
                        ].strip()

                        if piece:
                            chunks.append(
                                piece
                            )

                    subcurrent = ""

                else:

                    subcurrent = part

            if subcurrent:
                current = (
                    subcurrent
                )
            else:
                current = ""

        else:

            current = paragraph

    if current:
        chunks.append(
            current.strip()
        )

    return [
        chunk
        for chunk in chunks
        if chunk
    ]


# ============================================================
# CHUNK QUALITY
# ============================================================

def is_meaningful_chunk(
    text: str,
) -> bool:
    """
    Reject chunks that contain too little useful information.
    """

    stripped = text.strip()

    if len(stripped) < MIN_CHARS:
        return False

    alphanumeric = sum(
        char.isalnum()
        for char in stripped
    )

    if alphanumeric < 40:
        return False

    return True


# ============================================================
# CHUNK ID
# ============================================================

def make_chunk_id(
    document_id: str,
    index: int,
    content: str,
) -> str:
    """
    Generate deterministic chunk IDs.
    """

    digest = hashlib.sha1(
        content.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:12]

    return (
        f"{document_id}"
        f"-chunk-{index:04d}"
        f"-{digest}"
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def build_chunks_for_document(
    document: Dict,
) -> List[Dict]:
    """
    Convert one processed Marlin document into chunks.
    """

    document_id = str(
        document.get(
            "id",
            "unknown",
        )
    ).strip() or "unknown"

    title = str(
        document.get(
            "title",
            "",
        )
    ).strip()

    category = str(
        document.get(
            "category",
            "general",
        )
    ).strip() or "general"

    version = document.get(
        "version",
        None,
    )

    source_file = str(
        document.get(
            "source_file",
            "",
        )
    ).strip()

    source_url = str(
        document.get(
            "source_url",
            "",
        )
    ).strip()

    license_name = str(
        document.get(
            "license",
            "",
        )
    ).strip()

    raw_content = str(
        document.get(
            "content",
            "",
        )
    )

    content = clean_markdown(
        raw_content
    )

    if not content:
        return []

    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    document_type = detect_document_type(
        title,
        category,
        source_file,
    )

    # --------------------------------------------------------
    # DOCUMENT-LEVEL METADATA
    # --------------------------------------------------------

    document_gcode = (
        extract_gcode_commands(
            title,
            source_file,
            content,
        )
    )

    document_config = (
        extract_configuration_terms(
            content
        )
    )

    # --------------------------------------------------------
    # SECTION SPLITTING
    # --------------------------------------------------------

    sections = split_into_sections(
        content
    )

    # Extremely defensive fallback.
    #
    # Normally split_into_sections() should always return
    # something for non-empty content.
    if not sections:

        sections = [
            (
                "",
                content,
            )
        ]

    chunks: List[Dict] = []

    chunk_number = 0

    for heading_path, section_text in sections:

        section_chunks = split_large_text(
            section_text
        )

        for section_chunk in section_chunks:

            if not is_meaningful_chunk(
                section_chunk
            ):
                continue

            # ------------------------------------------------
            # SECTION LABEL
            # ------------------------------------------------

            #
            # Important:
            # Some documents contain content before their first
            # Markdown heading. In that case heading_path is empty.
            #
            # We therefore guarantee a useful section value.
            #
            section_label = (
                heading_path.strip()
                if heading_path
                and heading_path.strip()
                else (
                    title
                    if title
                    else "General"
                )
            )

            # ------------------------------------------------
            # CHUNK-SPECIFIC METADATA
            # ------------------------------------------------

            chunk_gcode = (
                extract_gcode_commands(
                    title,
                    source_file,
                    section_chunk,
                )
            )

            chunk_config = (
                extract_configuration_terms(
                    section_chunk
                )
            )

            # ------------------------------------------------
            # FALLBACK TO DOCUMENT-LEVEL METADATA
            # ------------------------------------------------

            #
            # If a chunk itself does not contain a code/config
            # symbol but the section is explicitly associated
            # with one, we do not blindly copy all document-level
            # metadata. This keeps retrieval precise.
            #

            chunk_number += 1

            chunk_id = make_chunk_id(
                document_id,
                chunk_number,
                section_chunk,
            )

            # ------------------------------------------------
            # RETRIEVAL TEXT
            # ------------------------------------------------

            retrieval_text_parts = [
                f"Document: {title}",
                f"Category: {category}",
                f"Type: {document_type}",
                f"Section: {section_label}",
            ]

            if chunk_gcode:

                retrieval_text_parts.append(
                    "G-code: "
                    + ", ".join(
                        chunk_gcode
                    )
                )

            if chunk_config:

                retrieval_text_parts.append(
                    "Configuration: "
                    + ", ".join(
                        chunk_config[:20]
                    )
                )

            retrieval_text_parts.append(
                section_chunk
            )

            retrieval_text = (
                "\n".join(
                    retrieval_text_parts
                ).strip()
            )

            # ------------------------------------------------
            # CHUNK RECORD
            # ------------------------------------------------

            chunk = {

                # ============================================
                # IDENTIFIERS
                # ============================================

                "chunk_id": chunk_id,
                "document_id": document_id,

                # ============================================
                # PRIMARY DOCUMENT METADATA
                # ============================================

                "title": title,
                "category": category,
                "document_type": document_type,
                "version": version,

                "source_file": source_file,
                "source_url": source_url,
                "license": license_name,

                "heading_path": section_label,

                # ============================================
                # TECHNICAL METADATA
                # ============================================

                "gcode_commands": chunk_gcode,
                "configuration_terms": chunk_config,

                # ============================================
                # CHUNK POSITION
                # ============================================

                "chunk_index": chunk_number,

                # ============================================
                # CONTENT
                # ============================================

                "content": section_chunk,

                "retrieval_text": retrieval_text,

                "content_length": len(
                    section_chunk
                ),

                # ============================================
                # RETRIEVAL / CITATION COMPATIBILITY
                # ============================================
                #
                # These aliases are intentionally included.
                #
                # Existing retrieval code may refer to:
                #     document
                #     source
                #     section
                #     gcode
                #     configuration
                #
                # Keeping both naming conventions makes the
                # metadata compatible with the current retriever
                # and future citation/UI components.
                #

                "document": title,
                "source": source_file,
                "section": section_label,
                "gcode": chunk_gcode,
                "configuration": chunk_config,
            }

            chunks.append(
                chunk
            )

    return chunks


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(
    path: Path,
) -> List[Dict]:
    """
    Load JSONL documents.
    """

    documents: List[Dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                documents.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                print(
                    f"WARNING: Invalid JSON "
                    f"on line {line_number}: "
                    f"{exc}"
                )

    return documents


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(
    chunks: List[Dict],
    path: Path,
) -> None:
    """
    Save chunks as JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# METADATA VALIDATION
# ============================================================

def validate_chunk_metadata(
    chunks: List[Dict],
) -> Dict[str, int]:
    """
    Validate important metadata fields.

    Returns counts for fields that are missing or empty.
    """

    stats = {
        "missing_document": 0,
        "missing_source": 0,
        "missing_section": 0,
        "missing_gcode": 0,
        "missing_configuration": 0,
    }

    for chunk in chunks:

        if not str(
            chunk.get(
                "document",
                "",
            )
        ).strip():

            stats[
                "missing_document"
            ] += 1

        if not str(
            chunk.get(
                "source",
                "",
            )
        ).strip():

            stats[
                "missing_source"
            ] += 1

        if not str(
            chunk.get(
                "section",
                "",
            )
        ).strip():

            stats[
                "missing_section"
            ] += 1

        # These are lists, so an empty list is valid.
        # We only check that the keys exist.
        if "gcode" not in chunk:
            stats[
                "missing_gcode"
            ] += 1

        if "configuration" not in chunk:
            stats[
                "missing_configuration"
            ] += 1

    return stats


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("MARLIN DOCUMENTATION - CHUNK BUILDER V2")
    print("=" * 80)

    print()
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    documents = load_documents(
        INPUT_FILE
    )

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    print()
    print("Building chunks...")
    print()

    all_chunks: List[Dict] = []

    document_chunk_counts: Dict[
        str,
        int
    ] = {}

    for index, document in enumerate(
        documents,
        start=1,
    ):

        title = (
            document.get(
                "title",
                "Unknown",
            )
            or "Unknown"
        )

        chunks = build_chunks_for_document(
            document
        )

        all_chunks.extend(
            chunks
        )

        document_chunk_counts[
            title
        ] = len(chunks)

        print(
            f"[{index:03d}/"
            f"{len(documents):03d}] "
            f"{title[:55]:55} "
            f"-> "
            f"{len(chunks):3d} chunks"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_chunks(
        all_chunks,
        OUTPUT_FILE,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    lengths = [
        chunk["content_length"]
        for chunk in all_chunks
    ]

    if lengths:

        average_length = (
            sum(lengths)
            / len(lengths)
        )

        maximum_length = max(
            lengths
        )

        minimum_length = min(
            lengths
        )

    else:

        average_length = 0
        maximum_length = 0
        minimum_length = 0

    gcode_chunks = sum(
        bool(
            chunk.get(
                "gcode_commands"
            )
        )
        for chunk in all_chunks
    )

    configuration_chunks = sum(
        bool(
            chunk.get(
                "configuration_terms"
            )
        )
        for chunk in all_chunks
    )

    # --------------------------------------------------------
    # METADATA VALIDATION
    # --------------------------------------------------------

    metadata_stats = (
        validate_chunk_metadata(
            all_chunks
        )
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)

    print(
        f"Documents       : "
        f"{len(documents)}"
    )

    print(
        f"Chunks          : "
        f"{len(all_chunks)}"
    )

    print(
        f"Average chars   : "
        f"{average_length:.0f}"
    )

    print(
        f"Minimum chars   : "
        f"{minimum_length}"
    )

    print(
        f"Maximum chars   : "
        f"{maximum_length}"
    )

    print(
        f"G-code chunks   : "
        f"{gcode_chunks}"
    )

    print(
        f"Config chunks   : "
        f"{configuration_chunks}"
    )

    print()
    print("-" * 80)
    print("METADATA VALIDATION")
    print("-" * 80)

    print(
        f"Missing document : "
        f"{metadata_stats['missing_document']}"
    )

    print(
        f"Missing source   : "
        f"{metadata_stats['missing_source']}"
    )

    print(
        f"Missing section  : "
        f"{metadata_stats['missing_section']}"
    )

    print(
        f"Missing gcode key: "
        f"{metadata_stats['missing_gcode']}"
    )

    print(
        f"Missing config key: "
        f"{metadata_stats['missing_configuration']}"
    )

    print()

    if any(
        value > 0
        for value in metadata_stats.values()
    ):

        print(
            "WARNING: Some metadata fields "
            "are missing."
        )

    else:

        print(
            "Metadata validation: PASSED"
        )

    print()

    print(
        f"Saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
