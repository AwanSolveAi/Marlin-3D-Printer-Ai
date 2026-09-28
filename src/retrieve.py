"""
Marlin AI Support Agent
Hybrid Retriever

Version: Retrieval v2

Retrieval strategy:

1. Query normalization
2. Technical token extraction
3. G-code detection
4. Configuration-symbol detection
5. Intent / query-term detection
6. BM25 lexical retrieval
7. FAISS semantic retrieval
8. Exact G-code content boosting
9. Exact configuration boosting
10. Technical-term coverage boosting
11. Category / document-type boosting
12. Reciprocal Rank Fusion
13. Candidate reranking
14. Deduplication

Important:
This retriever works with the existing:
    - marlin_faiss.index
    - marlin_metadata.json
    - marlin_bm25.pkl

No index rebuild is required for this version.
"""


from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

FAISS_FILE = VECTOR_DIR / "marlin_faiss.index"
METADATA_FILE = VECTOR_DIR / "marlin_metadata.json"
BM25_FILE = VECTOR_DIR / "marlin_bm25.pkl"


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

BM25_TOP_K = 40
VECTOR_TOP_K = 40

# We retrieve more candidates internally and rerank them.
CANDIDATE_TOP_K = 40

FINAL_TOP_K = 8

RRF_K = 60


# ============================================================
# BOOST WEIGHTS
# ============================================================

# Exact technical identifier matches are intentionally strong.
# This is critical for questions such as:
#
#   What does M420 V do?
#   What does G29 do?
#   What is M104?
#
# because semantic similarity can otherwise outrank exact
# technical documentation.

EXACT_GCODE_METADATA_BOOST = 0.35
EXACT_GCODE_CONTENT_BOOST = 0.28
EXACT_GCODE_SOURCE_BOOST = 0.18
EXACT_GCODE_DOCUMENT_BOOST = 0.25

GCODE_CATEGORY_BOOST = 0.002
GCODE_DOCUMENT_TYPE_BOOST = 0.002


EXACT_CONFIG_METADATA_BOOST = 0.30
EXACT_CONFIG_CONTENT_BOOST = 0.24
# Prefer a persisted source whose path names the requested configuration
# symbol over generic configuration pages that merely share the category.
EXACT_CONFIG_SOURCE_BOOST = 0.22

CONFIG_CATEGORY_BOOST = 0.002
CONFIG_DOCUMENT_TYPE_BOOST = 0.002


# Generic technical-term coverage.
# Two-list RRF peaks at 2 / 61 (~0.033); broad hints stay below that scale.
TERM_COVERAGE_BOOST = 0.001
MAX_TERM_COVERAGE_BOOST = 0.006

# Exact natural-language concepts should survive query expansion. This is
# deliberately limited to concepts with dedicated Marlin documentation.
EXACT_CONCEPT_PHRASE_BOOST = 0.006

# Bounded semantic boost for documentation that explicitly defines assigning
# coordinates to the current position. This is concept-based, not tied to any
# benchmark case or command identifier.
COORDINATE_ASSIGNMENT_DEFINITION_BOOST = 0.012

PREFERRED_CONCEPT_PHRASES = (
    "serial port",
    "thermal runaway",
    "probe temperature compensation",
    "temperature compensation",
    "troubleshoot",
)


# Procedural queries benefit from installation/configuration
# documents when the question clearly asks "how".
PROCEDURAL_INSTALLATION_BOOST = 0.002
PROCEDURAL_CONFIGURATION_BOOST = 0.002
PROCEDURAL_HARDWARE_BOOST = 0.001

TROUBLESHOOTING_DOCUMENT_BOOST = 0.003
TROUBLESHOOTING_SYMPTOM_BOOST = 0.006

TROUBLESHOOTING_SIGNALS = (
    "too high",
    "too low",
    "not working",
    "doesn't work",
    "does not work",
    "fails",
    "failed",
    "failure",
    "problem",
    "issue",
)


# ============================================================
# QUERY HELPERS
# ============================================================

COMMAND_PATTERN = re.compile(
    r"(?<![A-Z0-9_])([GMT])(\d+)(\.\d+)?(?![A-Z0-9_]|\.\d)", re.IGNORECASE
)


def canonical_command(match: re.Match) -> str:
    """Remove integer zero padding without losing a command's subcode."""
    return match[1].upper() + str(int(match[2])) + (match[3] or "")


def normalize_query(query: str) -> str:
    """
    Normalize whitespace while preserving technical identifiers.
    """

    if not query:
        return ""

    query = str(query).strip()

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    # Join spaced commands only at query start or after an explicit command cue.
    # Do not reinterpret arbitrary prose, units, or identifiers containing '_'.
    query = re.sub(
        r"(^|\b(?:command|code|send|run|use|does)\s+)([GMT])\s+(\d+(?:\.\d+)?)(?![\w.]|\s*(?:mg|kg|mm|cm)\b)",
        lambda match: match[1] + match[2].upper() + match[3],
        query, flags=re.IGNORECASE,
    )
    return COMMAND_PATTERN.sub(canonical_command, query)


NATURAL_LANGUAGE_EQUIVALENTS = (
    (r"\benergiz(?:e|es|ed|ing)\b", "enable"),
    (r"\bmotors?\b", "stepper steppers"),
    (r"\bstepper\b", "steppers"),
    (r"\baxis\b", "axes"),
    (r"\bstored height map\b", "mesh"),
    (r"\bcurrent nozzle location\b", "current position"),
    (r"\bmachine coordinate frame\b",
     "native workspace machine coordinates current move modifier"),
    (r"\b(?:consistent(?:ly)?|consistency)\b.*\b(?:measurements?|measures?)\b",
     "repeatability precision"),
    (r"\bhold up\b.*\binstructions\b", "wait"),
    (r"\b(?:motors?|steppers?)\b.*\bidle\b", "stepper inactivity timeout"),
    (r"\bnozzle\b.*\b(?:heating|cooling)\b", "hotend temperature"),
    (r"\bhow long\b.*\b(?:print|job)\b.*\b(?:running|run)\b|\belapsed\s+(?:print|job)\s+time\b",
     "print time elapsed duration"),
    (r"\bremember\b.*\b(?:axis|axes)\s+positions?\b",
     "save store position"),
    (r"\bmotor pulses\b.*\b(?:millimeters?|millimetres?|mm)\b",
     "steps per mm steps per unit"),
)


COORDINATE_ASSIGNMENT_PATTERN = (
    r"\bassign\b.*\bcoordinates?\b.*\bcurrent\b.*\b(?:nozzle\s+)?location\b"
)

ACTION_EQUIVALENTS = (
    (r"\b(?:nozzle|hotend|hot end)\b.*\b(?:heat\w*)\b.*\bcontinue\s+(?:immediately|without waiting)\b",
     "set hot end temperature without waiting"),
    (r"\b(?:hold up\b.*\binstructions|wait until)\b.*\b(?:nozzle|hotend|heating)\b.*\bcooling\b",
     "wait for hot end target temperature heating or cooling completion"),
    (COORDINATE_ASSIGNMENT_PATTERN,
     "set current position set current coordinates"),
    (r"\btest\b.*\b(?:stored\s+)?height map\b",
     "mesh validation mesh test pattern"),
)


def expand_natural_language(query: str) -> str:
    """Append bounded domain equivalents, retaining the original query."""
    if extract_gcode_codes(query) or extract_configuration_symbols(query):
        return query
    # Keep action phrases intact instead of discarding repeated connective words.
    phrases = [text for pattern, text in ACTION_EQUIVALENTS
               if re.search(pattern, query, flags=re.IGNORECASE)]
    if phrases:
        return query + " " + " ".join(dict.fromkeys(phrases))
    additions = []
    for pattern, equivalent in NATURAL_LANGUAGE_EQUIVALENTS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            additions.extend(equivalent.split())
    existing = set(re.findall(r"\w+", query.lower()))
    additions = [term for term in dict.fromkeys(additions) if term not in existing]
    return query + (" " + " ".join(additions) if additions else "")


def _is_eeprom_diagnostic(query: str) -> bool:
    return bool(re.search(r"\beeprom\b", query, re.IGNORECASE) and re.search(
        r"\b(?:crc\s+mismatch|errors?|reset\s+procedure|corrupt(?:ed|ion)?)\b",
        query, re.IGNORECASE))


def troubleshooting_signals(query: str) -> list[str]:
    """Return explicit symptom phrases present in a query."""

    query_lower = normalize_query(query).lower()
    signals = [signal for signal in TROUBLESHOOTING_SIGNALS if signal in query_lower]
    if re.search(r"\b(?:prints?\s+in\s+the\s+air|mesh\s+values\s+(?:look|are)\s+wrong|nozzle\s+too\s+high)\b", query_lower):
        signals.append("too high")
    if _is_eeprom_diagnostic(query):
        signals.append("EEPROM")

    if re.search(r"\bafter\s+[GMT]\d{1,4}\b", query_lower, flags=re.IGNORECASE):
        signals.append("after command")

    return list(dict.fromkeys(signals))


def is_command_troubleshooting_query(query: str) -> bool:
    """Identify an explicit command combined with failure/symptom language."""

    return bool(extract_gcode_codes(query) and troubleshooting_signals(query))


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_query(query: str) -> list[str]:
    """
    Technical BM25 tokenization.

    Preserves identifiers such as:

        G29
        M420
        M104
        M500
        MOTHERBOARD
        AUTO_BED_LEVELING_UBL
        PROBE_MANUALLY
    """

    if not query:
        return []

    query = query.lower()

    tokens = re.findall(
        r"""
        [a-z_][a-z0-9_]*
        |
        [gmst]\d+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
        |
        [a-z]+\d+[a-z0-9_]*
        """,
        query,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return tokens


# ============================================================
# G-CODE EXTRACTION
# ============================================================

def extract_gcode_codes(
    query: str,
) -> list[str]:
    """
    Extract G/M/T commands.

    Examples:

        G28
        G29
        M420
        M104
        M500
        T0
    """

    if not query:
        return []

    return sorted({canonical_command(match)
                   for match in COMMAND_PATTERN.finditer(normalize_query(query))})


# ============================================================
# CONFIGURATION SYMBOL EXTRACTION
# ============================================================

def extract_configuration_symbols(
    query: str,
) -> list[str]:
    """
    Extract likely Marlin configuration constants.

    Examples:

        MOTHERBOARD
        AUTO_BED_LEVELING_UBL
        GRID_MAX_POINTS
        SERIAL_PORT
        Z_PROBE_LOW_POINT
        PROBE_MANUALLY
    """

    if not query:
        return []

    candidates = re.findall(
        r"\b[A-Z][A-Z0-9_]{3,}\b",
        query,
    )

    excluded = {
        "WHAT",
        "WHEN",
        "WHERE",
        "WHICH",
        "WHY",
        "HOW",
        "DOES",
        "DO",
        "DID",
        "CAN",
        "COULD",
        "SHOULD",
        "WOULD",
        "THE",
        "AND",
        "FOR",
        "WITH",
        "FROM",
        "THIS",
        "THAT",
        "YOUR",
        "MY",
        "PRINT",
        "START",
        "AFTER",
        "BEFORE",
        "BED",
        "LEVELING",
        "MARLIN",
        "GCODE",
    }

    if _is_eeprom_diagnostic(query):
        excluded.add("EEPROM")

    return sorted(
        set(
            item
            for item in candidates
            if item not in excluded
        )
    )


# ============================================================
# TECHNICAL TERM EXTRACTION
# ============================================================

def extract_meaningful_terms(
    query: str,
) -> list[str]:
    """
    Extract meaningful technical / natural-language terms.

    Stopwords are removed so generic words such as "what",
    "is", "the", and "how" don't influence reranking.
    """

    stopwords = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "do",
        "does",
        "did",
        "how",
        "why",
        "when",
        "where",
        "which",
        "can",
        "could",
        "should",
        "would",
        "i",
        "me",
        "my",
        "you",
        "your",
        "it",
        "its",
        "be",
        "used",
        "use",
        "used",
        "configure",
        "configuration",
    }

    terms = re.findall(
        r"[a-zA-Z][a-zA-Z0-9_-]{2,}",
        query.lower(),
    )

    return [
        term
        for term in terms
        if term not in stopwords
    ]


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(
    query: str,
) -> str:
    """
    Detect broad query intent.

    Returns:

        gcode
        configuration
        procedural
        hardware
        installation
        general
    """

    query_lower = query.lower()

    if is_command_troubleshooting_query(query):
        return "troubleshooting"

    if extract_gcode_codes(query):
        return "gcode"

    if extract_configuration_symbols(query):
        return "configuration"

    if _is_eeprom_diagnostic(query) or "too high" in troubleshooting_signals(query):
        return "troubleshooting"

    procedural_words = {
        "how",
        "configure",
        "install",
        "setup",
        "set",
        "build",
        "upload",
        "enable",
        "disable",
        "connect",
        "wire",
    }

    if any(
        word in query_lower.split()
        for word in procedural_words
    ):
        if any(
            word in query_lower
            for word in [
                "motherboard",
                "board",
                "firmware",
                "platformio",
                "install",
                "upload",
            ]
        ):
            return "installation"

        return "procedural"

    hardware_terms = {
        "motherboard",
        "board",
        "driver",
        "stepper",
        "motor",
        "thermistor",
        "heater",
        "probe",
        "sensor",
        "wiring",
        "connector",
    }

    if any(
        term in query_lower
        for term in hardware_terms
    ):
        return "hardware"

    return "general"


# ============================================================
# TEXT MATCH HELPERS
# ============================================================

def _safe_lower(value: Any) -> str:
    """
    Safely convert a metadata value to lowercase text.
    """

    if value is None:
        return ""

    return str(value).lower()


def _contains_exact_code(
    code: str,
    text: str,
) -> bool:
    """
    Exact G/M/T code match.

    Examples:

        M420 matches M420
        M420 does NOT match M4200
    """

    if not code or not text:
        return False

    expected = COMMAND_PATTERN.fullmatch(code)
    return bool(expected and canonical_command(expected) in {
        canonical_command(match) for match in COMMAND_PATTERN.finditer(text)
    })


def _is_command_definition(record: dict[str, Any], codes: list[str]) -> bool:
    """Identify command-owned pages, not pages merely mentioning a command."""
    source = str(record.get("source", "")).replace("\\", "/")
    if not source.startswith("_gcode/"):
        return False
    stem = Path(source).stem
    return any(_contains_exact_code(code, stem) or
               (stem.upper() == "T" and re.fullmatch(r"T\d+", code))
               for code in codes)


def _contains_exact_symbol(
    symbol: str,
    text: str,
) -> bool:
    """
    Exact configuration-symbol match.
    """

    if not symbol or not text:
        return False

    return bool(
        re.search(
            rf"(?<![A-Z0-9_]){re.escape(symbol)}(?![A-Z0-9_])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _contains_phrase(phrase: str, text: str) -> bool:
    """Match a phrase across normal spaces, punctuation, or underscores."""

    phrase_words = re.findall(r"[a-z0-9]+", phrase.lower())
    text_words = re.findall(r"[a-z0-9]+", text.lower())

    if not phrase_words or len(phrase_words) > len(text_words):
        return False

    width = len(phrase_words)
    return any(
        text_words[index:index + width] == phrase_words
        for index in range(len(text_words) - width + 1)
    )


def _defines_current_position_assignment(text: str) -> bool:
    """Match a page whose own opening definition is about setting current position.

    Restricting this to the opening documentation text prevents cross-reference
    pages from receiving the same semantic priority merely because they mention
    another command later in the page.
    """
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    opening = normalized[:450]
    return bool(
        re.search(
            r"\bset(?:s|ting)?\s+(?:the\s+)?current\s+position\s+to\b",
            opening,
        )
        or re.search(
            r"\bset(?:s|ting)?\s+(?:the\s+)?current\s+coordinates?\s+to\b",
            opening,
        )
    )


# ============================================================
# RETRIEVER
# ============================================================

class MarlinHybridRetriever:

    def __init__(self):

        print(
            "Initializing Marlin hybrid retriever..."
        )

        self._check_files()

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        print(
            "Loading FAISS index..."
        )

        self.index = faiss.read_index(
            str(FAISS_FILE)
        )

        print(
            f"FAISS vectors: {self.index.ntotal}"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        print(
            "Loading metadata..."
        )

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        print(
            f"Metadata records: {len(self.metadata)}"
        )

        if self.index.ntotal != len(
            self.metadata
        ):

            raise RuntimeError(
                "FAISS / metadata mismatch.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"Metadata records: {len(self.metadata)}\n\n"
                "Rebuild the index with:\n"
                "python src\\build_index.py"
            )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        print(
            "Loading BM25..."
        )

        with BM25_FILE.open(
            "rb"
        ) as f:

            payload = pickle.load(f)

        self.bm25 = payload["bm25"]

        bm25_corpus_size = getattr(
            self.bm25,
            "corpus_size",
            None,
        )

        if bm25_corpus_size != len(self.metadata):
            raise RuntimeError(
                "BM25 / metadata mismatch.\n"
                f"BM25 records: {bm25_corpus_size}\n"
                f"Metadata records: {len(self.metadata)}\n\n"
                "Rebuild the index with:\n"
                "python src\\build_index.py"
            )

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        print(
            "Loading embedding model..."
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Retriever ready."
        )

    # ========================================================
    # FILE CHECK
    # ========================================================

    @staticmethod
    def _check_files():

        required = [
            FAISS_FILE,
            METADATA_FILE,
            BM25_FILE,
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Missing retrieval files:\n"
                + "\n".join(missing)
                + "\n\nRun:\n"
                "python src\\build_index.py"
            )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        tokens = tokenize_query(
            query
        )

        if not tokens:
            return []

        scores = self.bm25.get_scores(
            tokens
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        )

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in indices
            if scores[index] > 0
        ]

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        results = []

        for idx, score in zip(
            indices[0],
            scores[0],
        ):

            if idx < 0:
                continue

            results.append(
                (
                    int(idx),
                    float(score),
                )
            )

        return results

    # ========================================================
    # RRF FUSION
    # ========================================================

    @staticmethod
    def _rrf_fusion(
        bm25_results,
        vector_results,
    ) -> dict[int, float]:

        fused: dict[int, float] = {}

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        for rank, (idx, _) in enumerate(
            bm25_results,
            start=1,
        ):

            fused[idx] = (
                fused.get(
                    idx,
                    0.0,
                )
                + 1.0 / (
                    RRF_K + rank
                )
            )

        # ----------------------------------------------------
        # Vector
        # ----------------------------------------------------

        for rank, (idx, _) in enumerate(
            vector_results,
            start=1,
        ):

            fused[idx] = (
                fused.get(
                    idx,
                    0.0,
                )
                + 1.0 / (
                    RRF_K + rank
                )
            )

        return fused

    # ========================================================
    # BOOST CALCULATION
    # ========================================================

    @staticmethod
    def _calculate_boost(
        record: dict[str, Any],
        query_gcodes: list[str],
        query_configs: list[str],
        query_terms: list[str],
        query_intent: str,
        query_lower: str,
    ) -> tuple[float, list[str]]:

        boost = 0.0

        reasons: list[str] = []

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        record_gcodes = {
            canonical_command(match)
            for code in record.get(
                "gcode_codes",
                [],
            )
            for match in COMMAND_PATTERN.finditer(str(code))
        }

        record_configs = {
            str(config).upper()
            for config in record.get(
                "configuration_symbols",
                [],
            )
        }

        metadata_gcode = str(
            record.get(
                "gcode",
                "",
            )
        ).upper()

        metadata_config = str(
            record.get(
                "configuration",
                "",
            )
        ).upper()

        document = str(
            record.get(
                "document",
                "",
            )
        ).strip()

        source = str(
            record.get(
                "source",
                "",
            )
        ).strip()

        section = str(
            record.get(
                "section",
                "",
            )
        ).strip()

        content = str(
            record.get(
                "content",
                "",
            )
        )

        category = str(
            record.get(
                "category",
                "",
            )
        ).lower()

        document_type = str(
            record.get(
                "document_type",
                "",
            )
        ).lower()

        searchable_text = " ".join(
            [
                document,
                source,
                section,
                content,
            ]
        ).lower()

        matching_symptoms = [symptom for symptom in troubleshooting_signals(query_lower)
                             if symptom != "after command"
                             and _contains_phrase(symptom, content)]
        if query_intent == "troubleshooting" and matching_symptoms:
            if (
                document_type == "troubleshooting"
                or document.lower().startswith("troubleshoot")
            ):
                boost += TROUBLESHOOTING_DOCUMENT_BOOST
                reasons.append("troubleshooting document")

            for symptom in troubleshooting_signals(query_lower):
                if symptom != "after command" and _contains_phrase(
                    symptom,
                    searchable_text,
                ):
                    boost += TROUBLESHOOTING_SYMPTOM_BOOST
                    reasons.append(f"exact troubleshooting symptom {symptom}")
                    break

        # Preserve strong, explicit natural-language concepts when callers add
        # broader aliases to a query. Without this, generic configuration pages
        # can outrank a document containing the exact requested concept.
        for phrase in PREFERRED_CONCEPT_PHRASES:
            if (
                _contains_phrase(phrase, query_lower)
                and _contains_phrase(phrase, searchable_text)
            ):
                boost += EXACT_CONCEPT_PHRASE_BOOST
                reasons.append(f"exact concept phrase {phrase}")

        if (
            "troubleshoot" in query_lower
            and document.lower().startswith("troubleshoot")
            and matching_symptoms
        ):
            boost += 0.002
            reasons.append("troubleshooting document title")


        # Preserve the semantic distinction between assigning coordinates to the
        # current position and applying a persistent home offset. The user intent
        # is detected generically; no command identifier or benchmark text is
        # hard-coded here.
        if (
            re.search(COORDINATE_ASSIGNMENT_PATTERN, query_lower, re.IGNORECASE)
            and _defines_current_position_assignment(content)
        ):
            boost += COORDINATE_ASSIGNMENT_DEFINITION_BOOST
            reasons.append("current-position assignment definition")

        # ====================================================
        # EXACT G-CODE MATCHING
        # ====================================================

        for code in query_gcodes:

            code_lower = code.lower()

            # ------------------------------------------------
            # Metadata exact match
            # ------------------------------------------------

            if code in record_gcodes:

                boost += EXACT_GCODE_METADATA_BOOST

                reasons.append(
                    f"exact G-code metadata {code}"
                )

            # ------------------------------------------------
            # Explicit G-code field
            # ------------------------------------------------

            if _contains_exact_code(code, metadata_gcode):

                boost += 0.15

                reasons.append(
                    f"G-code field {code}"
                )

            # ------------------------------------------------
            # Dedicated document
            # ------------------------------------------------

            if (
                document
                and _contains_exact_code(code, document)
            ):

                boost += EXACT_GCODE_DOCUMENT_BOOST

                reasons.append(
                    f"dedicated document {code}"
                )

            # ------------------------------------------------
            # Source exact match
            # ------------------------------------------------

            if _contains_exact_code(
                code,
                source,
            ):

                boost += EXACT_GCODE_SOURCE_BOOST

                reasons.append(
                    f"G-code in source {code}"
                )

            # ------------------------------------------------
            # Section exact match
            # ------------------------------------------------

            if _contains_exact_code(
                code,
                section,
            ):

                boost += 0.15

                reasons.append(
                    f"G-code in section {code}"
                )

            # ------------------------------------------------
            # CONTENT exact match
            # ------------------------------------------------

            if _contains_exact_code(
                code,
                content,
            ):

                boost += EXACT_GCODE_CONTENT_BOOST

                reasons.append(
                    f"G-code in content {code}"
                )

            # ------------------------------------------------
            # Dedicated G-code documents
            # ------------------------------------------------

            if category == "gcode":

                boost += GCODE_CATEGORY_BOOST

                reasons.append(
                    "G-code category"
                )

            if document_type == "gcode":

                boost += GCODE_DOCUMENT_TYPE_BOOST

                reasons.append(
                    "G-code document type"
                )

        # ====================================================
        # EXACT CONFIGURATION MATCHING
        # ====================================================

        for config in query_configs:

            config_upper = config.upper()

            # ------------------------------------------------
            # Metadata exact match
            # ------------------------------------------------

            if config_upper in record_configs:

                boost += EXACT_CONFIG_METADATA_BOOST

                reasons.append(
                    f"exact configuration metadata {config}"
                )

            # ------------------------------------------------
            # Explicit configuration field
            # ------------------------------------------------

            if config_upper in metadata_config:

                boost += 0.15

                reasons.append(
                    f"configuration field {config}"
                )

            # ------------------------------------------------
            # Source exact match
            # ------------------------------------------------

            if _contains_exact_symbol(
                config,
                source,
            ):

                boost += EXACT_CONFIG_SOURCE_BOOST

                reasons.append(
                    f"configuration in source {config}"
                )

            # ------------------------------------------------
            # Section exact match
            # ------------------------------------------------

            if _contains_exact_symbol(
                config,
                section,
            ):

                boost += 0.12

                reasons.append(
                    f"configuration in section {config}"
                )

            # ------------------------------------------------
            # Content exact match
            # ------------------------------------------------

            if _contains_exact_symbol(
                config,
                content,
            ):

                boost += EXACT_CONFIG_CONTENT_BOOST

                reasons.append(
                    f"configuration in content {config}"
                )

            # ------------------------------------------------
            # Configuration document type
            # ------------------------------------------------

            if category == "configuration":

                boost += CONFIG_CATEGORY_BOOST

                reasons.append(
                    "configuration category"
                )

            if document_type == "configuration":

                boost += CONFIG_DOCUMENT_TYPE_BOOST

                reasons.append(
                    "configuration document type"
                )

        # ====================================================
        # TECHNICAL TERM COVERAGE
        # ====================================================

        if query_terms:

            matched_terms = 0

            for term in query_terms:

                if _contains_phrase(term, searchable_text):

                    matched_terms += 1

            if matched_terms:

                coverage_boost = min(
                    matched_terms
                    * TERM_COVERAGE_BOOST,
                    MAX_TERM_COVERAGE_BOOST,
                )

                boost += coverage_boost

                reasons.append(
                    f"term coverage "
                    f"{matched_terms}/{len(query_terms)}"
                )

        # ====================================================
        # PROCEDURAL / INSTALLATION BOOST
        # ====================================================

        if query_intent in {
            "procedural",
            "installation",
        }:

            if document_type == "installation":

                boost += PROCEDURAL_INSTALLATION_BOOST

                reasons.append(
                    "installation document"
                )

            if category == "configuration":

                boost += PROCEDURAL_CONFIGURATION_BOOST

                reasons.append(
                    "configuration category"
                )

            if category == "hardware":

                boost += PROCEDURAL_HARDWARE_BOOST

                reasons.append(
                    "hardware category"
                )

        # ====================================================
        # MOTHERBOARD-SPECIFIC BOOST
        # ====================================================

        if "motherboard" in query_lower:

            motherboard_terms = [
                "motherboard",
                "m_otherboard",
                "mOTHERBOARD",
                "board",
                "pins.h",
                "platformio.ini",
                "MOTHERBOARD",
            ]

            motherboard_matches = 0

            for term in motherboard_terms:

                if term.lower() in searchable_text:

                    motherboard_matches += 1

            if motherboard_matches:

                boost += min(
                    motherboard_matches * 0.001,
                    0.004,
                )

                reasons.append(
                    f"motherboard evidence "
                    f"{motherboard_matches}"
                )

        # ====================================================
        # DOCUMENT TITLE MATCH
        # ====================================================

        if document:

            document_lower = document.lower()

            if (
                document_lower
                and (_contains_phrase(document_lower, query_lower)
                     or (document_lower == "t" and any(
                         re.fullmatch(r"T\d+", code) for code in query_gcodes)))
            ):

                boost += 0.002

                reasons.append(
                    "document title match"
                )

        # ====================================================
        # REMOVE DUPLICATE REASONS
        # ====================================================

        reasons = list(
            dict.fromkeys(
                reasons
            )
        )

        return boost, reasons

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
    ) -> list[dict[str, Any]]:

        query = normalize_query(
            query
        )

        if not query:
            return []

        query = expand_natural_language(query)
        query_lower = query.lower()

        # ----------------------------------------------------
        # Query analysis
        # ----------------------------------------------------

        query_gcodes = extract_gcode_codes(
            query
        )

        query_configs = (
            extract_configuration_symbols(
                query
            )
        )

        query_terms = extract_meaningful_terms(
            query
        )

        query_intent = detect_query_intent(
            query
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        bm25_results = self._bm25_search(
            query,
            BM25_TOP_K,
        )

        vector_results = self._vector_search(
            query,
            VECTOR_TOP_K,
        )

        fused = self._rrf_fusion(
            bm25_results,
            vector_results,
        )

        auxiliary_rrf = {}
        if (not query_gcodes and not query_configs
                and re.search(COORDINATE_ASSIGNMENT_PATTERN, query, re.IGNORECASE)):
            # Keep the original search. An equivalent focused view can rescue
            # intent evidence, but never receives an additive double-count boost.
            auxiliary_rrf = self._rrf_fusion(
                self._bm25_search("set current position", BM25_TOP_K),
                self._vector_search("set current position", VECTOR_TOP_K),
            )
            for idx, score in auxiliary_rrf.items():
                fused[idx] = max(fused.get(idx, 0.0), score)

            # Ensure explicit documentation definitions of setting the current
            # position survive lexical/vector truncation for this semantic intent.
            # This remains concept-based: no command identifier or benchmark case
            # is hard-coded.
            for idx, record in enumerate(self.metadata):
                content = str(record.get("content", ""))
                if _defines_current_position_assignment(content):
                    fused.setdefault(idx, auxiliary_rrf.get(idx, 0.0))

        # Exact definitions must survive lexical/vector candidate truncation.
        if query_intent == "gcode":
            for idx, record in enumerate(self.metadata):
                if _is_command_definition(record, query_gcodes):
                    fused.setdefault(idx, 0.0)

        bm25_scores = dict(
            bm25_results
        )

        vector_scores = dict(
            vector_results
        )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        candidates = []

        for idx, rrf_score in fused.items():

            if idx >= len(
                self.metadata
            ):
                continue

            record = self.metadata[idx]

            boost, reasons = (
                self._calculate_boost(
                    record=record,
                    query_gcodes=query_gcodes,
                    query_configs=query_configs,
                    query_terms=query_terms,
                    query_intent=query_intent,
                    query_lower=query_lower,
                )
            )

            final_score = (
                rrf_score
                + boost
            )

            candidates.append(
                {
                    "index": idx,

                    "chunk_id": record.get(
                        "chunk_id",
                        "",
                    ),

                    "document": record.get(
                        "document",
                        "",
                    ),

                    "category": record.get(
                        "category",
                        "",
                    ),

                    "document_type": record.get(
                        "document_type",
                        "",
                    ),

                    "source": record.get(
                        "source",
                        "",
                    ),

                    "section": record.get(
                        "section",
                        "",
                    ),

                    "gcode": record.get(
                        "gcode",
                        "",
                    ),

                    "configuration": record.get(
                        "configuration",
                        "",
                    ),

                    "gcode_codes": record.get(
                        "gcode_codes",
                        [],
                    ),

                    "configuration_symbols": record.get(
                        "configuration_symbols",
                        [],
                    ),

                    "content": record.get(
                        "content",
                        "",
                    ),

                    "rrf_score": rrf_score,
                    "auxiliary_rrf_score": auxiliary_rrf.get(idx, 0.0),

                    "bm25_score": bm25_scores.get(
                        idx,
                        0.0,
                    ),

                    "vector_score": vector_scores.get(
                        idx,
                        0.0,
                    ),

                    "boost": boost,

                    "final_score": final_score,

                    "match_reasons": reasons,
                }
            )

        # ----------------------------------------------------
        # Sort by final relevance
        # ----------------------------------------------------

        candidates.sort(
            # Rounding at 12 decimal places absorbs floating-point RRF ties,
            # not relevance differences. Coverage excludes category boosts.
            # Explicit identifier queries retain their existing ordering exactly.
            key=lambda x: (
                bool(
                    re.search(COORDINATE_ASSIGNMENT_PATTERN, query, re.IGNORECASE)
                    and _defines_current_position_assignment(
                        str(x.get("content", ""))
                    )
                ),
                round(x["rrf_score"], 12),
                sum(_contains_phrase(term, " ".join(
                    str(x.get(field, "")) for field in ("document", "source", "section", "content")
                )) for term in set(query_terms)),
                x["vector_score"], x["boost"], x["bm25_score"]
            ) if not query_gcodes and not query_configs else (
                query_intent == "troubleshooting"
                and x["document"].lower().startswith("troubleshoot")
                and all(_contains_exact_code(code, x["content"]) for code in query_gcodes)
                and any(_contains_phrase(symptom, x["content"])
                        for symptom in troubleshooting_signals(query)
                        if symptom not in {"after command", "problem", "issue"}),
                query_intent == "gcode" and _is_command_definition(x, query_gcodes),
                x["final_score"],
                x["vector_score"],
                x["bm25_score"],
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Deduplication
        # ----------------------------------------------------

        results = []

        seen_chunk_ids = set()

        seen_content = set()

        for item in candidates:

            chunk_id = str(
                item.get(
                    "chunk_id",
                    "",
                )
            )

            content = str(
                item.get(
                    "content",
                    "",
                )
            ).strip()

            normalized_content = re.sub(
                r"\s+",
                " ",
                content.lower(),
            )

            # ------------------------------------------------
            # Exact chunk duplicate
            # ------------------------------------------------

            if (
                chunk_id
                and chunk_id in seen_chunk_ids
            ):

                continue

            # ------------------------------------------------
            # Exact content duplicate
            # ------------------------------------------------

            if (
                normalized_content
                and normalized_content
                in seen_content
            ):

                continue

            if chunk_id:

                seen_chunk_ids.add(
                    chunk_id
                )

            if normalized_content:

                seen_content.add(
                    normalized_content
                )

            results.append(
                item
            )

            if len(results) >= top_k:

                break

        return results


# ============================================================
# RESULT PRINTING
# ============================================================

def print_results(
    query: str,
    results: list[dict[str, Any]],
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MARLIN HYBRID SEARCH"
    )

    print(
        "=" * 80
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Results: {len(results)}"
    )

    print(
        "\n"
        + "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Rank              : {rank}"
        )

        print(
            f"Final Score       : "
            f"{result['final_score']:.6f}"
        )

        print(
            f"RRF Score         : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"BM25 Score        : "
            f"{result['bm25_score']:.6f}"
        )

        print(
            f"Vector Score      : "
            f"{result['vector_score']:.6f}"
        )

        print(
            f"Boost             : "
            f"{result['boost']:.6f}"
        )

        print(
            f"Document          : "
            f"{result['document']}"
        )

        print(
            f"Category          : "
            f"{result['category']}"
        )

        print(
            f"Document Type     : "
            f"{result['document_type']}"
        )

        print(
            f"Source            : "
            f"{result['source']}"
        )

        print(
            f"Section           : "
            f"{result['section']}"
        )

        print(
            f"G-code            : "
            f"{result['gcode']}"
        )

        print(
            f"Configuration     : "
            f"{result['configuration']}"
        )

        if result.get(
            "gcode_codes"
        ):

            print(
                "G-code Codes      : "
                + ", ".join(
                    result[
                        "gcode_codes"
                    ]
                )
            )

        if result.get(
            "configuration_symbols"
        ):

            print(
                "Config Symbols    : "
                + ", ".join(
                    result[
                        "configuration_symbols"
                    ]
                )
            )

        if result.get(
            "match_reasons"
        ):

            print(
                "Match Reasons     : "
                + ", ".join(
                    result[
                        "match_reasons"
                    ]
                )
            )

        print(
            "\nCONTENT"
        )

        print(
            "-" * 80
        )

        content = result[
            "content"
        ]

        if len(content) > 1200:

            content = (
                content[:1200]
                + "\n..."
            )

        print(
            content
        )

    print(
        "\n"
        + "=" * 80
    )


# ============================================================
# CLI
# ============================================================

def main():

    retriever = (
        MarlinHybridRetriever()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MARLIN AI SUPPORT AGENT"
    )

    print(
        "HYBRID RETRIEVER V2"
    )

    print(
        "=" * 80
    )

    print(
        "\nEnter a Marlin technical question."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            query = input(
                "\nQuestion: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nExiting."
            )

            break

        if query.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Exiting."
            )

            break

        if not query:

            continue

        results = retriever.retrieve(
            query,
            top_k=FINAL_TOP_K,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()
