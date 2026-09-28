from pathlib import Path
import json
import re
import hashlib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "MarlinDocumentation"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "marlin_documents.jsonl"


# ============================================================
# DOCUMENTATION CATEGORIES
# ============================================================

CATEGORY_MAP = {
    "_basics": "basics",
    "_configuration": "configuration",
    "_data": "data",
    "_development": "development",
    "_features": "features",
    "_gcode": "gcode",
    "_hardware": "hardware",
    "_setting": "settings",
    "_tools": "tools",
}


# ============================================================
# DIRECTORIES TO EXCLUDE
# ============================================================

EXCLUDED_DIRECTORIES = {
    ".git",
    ".github",
    ".well-known",
    "assets",
    "feeds",
    "_includes",
    "_layouts",
    "_meta",
    "_plugins",
    "_sass",
    "_tmp",
}


# ============================================================
# FILES TO EXCLUDE
# ============================================================

EXCLUDED_FILES = {
    "README.md",
    "donate.md",
    "download.md",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraphs.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():
        line = line.rstrip()

        # Remove excessive spaces
        line = re.sub(r"[ \t]+", " ", line)

        lines.append(line)

    text = "\n".join(lines)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_front_matter(text: str) -> str:
    """
    Remove Jekyll YAML front matter.
    """

    text = text.lstrip()

    if not text.startswith("---"):
        return text

    match = re.match(
        r"^---\s*\n.*?\n---\s*\n",
        text,
        flags=re.DOTALL,
    )

    if match:
        text = text[match.end():]

    return text.strip()


def remove_html_comments(text: str) -> str:
    """
    Remove HTML comments.
    """

    return re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )


def remove_jekyll_liquid(text: str) -> str:
    """
    Remove Jekyll Liquid tags and expressions.
    """

    text = re.sub(
        r"\{\{.*?\}\}",
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"\{%.*?%\}",
        "",
        text,
        flags=re.DOTALL,
    )

    return text


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags while preserving their text content.

    Important:
    Markdown code blocks are preserved because we only
    remove HTML tags outside fenced code blocks.
    """

    parts = re.split(
        r"(```.*?```)",
        text,
        flags=re.DOTALL,
    )

    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"<[^>]+>",
            "",
            parts[i],
        )

    return "".join(parts)


def clean_markdown_links(text: str) -> str:
    """
    Preserve link text while removing unnecessary
    relative/HTML attributes.

    Example:

        [Install Marlin](/docs/install.html)

    becomes:

        Install Marlin
    """

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Remove Jekyll target attributes
    text = re.sub(
        r"\{:target=\"[^\"]+\"\}",
        "",
        text,
    )

    # Remove common inline Jekyll classes
    text = re.sub(
        r"\{:\s*[^}]+\}",
        "",
        text,
    )

    return text


def clean_image_markdown(text: str) -> str:
    """
    Remove image syntax.

    Images are not currently part of our text RAG pipeline.
    """

    return re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        "",
        text,
    )


def clean_document(text: str) -> str:
    """
    Clean a Marlin documentation document.

    Technical information such as:

    - G-code commands
    - configuration names
    - code blocks
    - parameters
    - headings

    is intentionally preserved.
    """

    text = remove_front_matter(text)

    text = remove_html_comments(text)

    text = remove_jekyll_liquid(text)

    text = clean_image_markdown(text)

    text = clean_markdown_links(text)

    text = remove_html_tags(text)

    text = normalize_whitespace(text)

    return text


def extract_title(text: str, path: Path) -> str:
    """
    Extract the first Markdown H1 heading.

    If no H1 exists, use the filename.
    """

    match = re.search(
        r"^\s*#\s+(.+?)\s*$",
        text,
        flags=re.MULTILINE,
    )

    if match:
        title = match.group(1).strip()

        title = re.sub(
            r"`([^`]+)`",
            r"\1",
            title,
        )

        return title

    return (
        path.stem
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )


def detect_category(path: Path) -> str:
    """
    Detect category from the Marlin documentation directory.
    """

    try:
        relative = path.relative_to(RAW_DIR)
    except ValueError:
        return "unknown"

    for part in relative.parts:

        if part in CATEGORY_MAP:
            return CATEGORY_MAP[part]

    return "general"


def extract_version(path: Path) -> str | None:
    """
    Detect documentation version directories.

    Examples:

        1.1
        2.0.9
    """

    for part in path.parts:

        if re.fullmatch(
            r"\d+\.\d+(?:\.\d+)?",
            part,
        ):
            return part

    return None


def build_source_url(path: Path) -> str:
    """
    Build the public Marlin documentation URL.
    """

    try:
        relative = path.relative_to(RAW_DIR)
    except ValueError:
        return "https://marlinfw.org/"

    parts = list(relative.parts)

    clean_parts = []

    for part in parts:

        if part.startswith("_"):
            part = part[1:]

        clean_parts.append(part)

    filename = clean_parts[-1]

    if filename.lower().endswith(".md"):
        filename = filename[:-3]

    elif filename.lower().endswith(".html"):
        filename = filename[:-5]

    clean_parts[-1] = filename

    return (
        "https://marlinfw.org/docs/"
        + "/".join(clean_parts)
        + ".html"
    )


def generate_document_id(path: Path) -> str:
    """
    Generate a stable document identifier.
    """

    relative = str(
        path.relative_to(RAW_DIR)
    ).replace("\\", "/")

    digest = hashlib.sha1(
        relative.encode("utf-8")
    ).hexdigest()[:12]

    return f"marlin-{digest}"


def is_excluded(path: Path) -> bool:
    """
    Determine whether a file should be excluded.
    """

    if path.name in EXCLUDED_FILES:
        return True

    for part in path.parts:

        if part in EXCLUDED_DIRECTORIES:
            return True

    return False


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_file(path: Path) -> dict | None:
    """
    Process one Markdown document.
    """

    try:

        raw_text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception as exc:

        print(
            f"[WARNING] Could not read {path}: {exc}"
        )

        return None

    cleaned_text = clean_document(
        raw_text
    )

    if not cleaned_text:
        return None

    title = extract_title(
        cleaned_text,
        path,
    )

    category = detect_category(
        path
    )

    version = extract_version(
        path
    )

    relative_path = str(
        path.relative_to(RAW_DIR)
    ).replace("\\", "/")

    document = {

        "id": generate_document_id(path),

        "title": title,

        "category": category,

        "version": version,

        "source_file": relative_path,

        "source_url": build_source_url(path),

        "license": "GPL-3.0",

        "content_type": "technical_documentation",

        "content": cleaned_text,
    }

    return document


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 80)
    print("MARLIN DOCUMENTATION INGESTION")
    print("=" * 80)

    print(
        f"\nSource directory:\n{RAW_DIR}"
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Marlin documentation not found:\n{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = list(
        RAW_DIR.rglob("*.md")
    )

    print(
        f"\nMarkdown files found: {len(files)}"
    )

    processed = 0
    skipped = 0

    category_counts = {}

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for path in files:

            if is_excluded(path):

                skipped += 1

                continue

            document = process_file(
                path
            )

            if document is None:

                skipped += 1

                continue

            output.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed += 1

            category = document[
                "category"
            ]

            category_counts[
                category
            ] = (
                category_counts.get(
                    category,
                    0
                )
                + 1
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nProcessed documents : {processed}"
    )

    print(
        f"Skipped files       : {skipped}"
    )

    print(
        "\nDocuments by category:"
    )

    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  {category:<20} {count}"
        )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()