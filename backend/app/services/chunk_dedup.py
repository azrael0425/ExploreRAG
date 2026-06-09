"""
Pre-ingestion Deduplication Pipeline
=====================================

Filters noise and removes duplicate/near-duplicate chunks BEFORE embedding,
reducing vector space pollution and improving retrieval quality.

Three-stage pipeline:
  1. Noise filter  — remove boilerplate headers/footers, legal junk, tiny chunks
  2. Exact dedup   — SHA-256 content hash to drop identical chunks
  3. Near dedup    — character n-gram shingling + Jaccard similarity for fuzzy matches
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

from app.core.config import settings
from app.services.models.parsed_document import EnrichedChunk

logger = logging.getLogger(__name__)

# ── Compiled boilerplate patterns ────────────────────────────────────────
# Each pattern matches a FULL chunk that is predominantly boilerplate.
# We use re.IGNORECASE | re.DOTALL so multiline chunks are handled.

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Copyright / license lines
    re.compile(
        r"^[\s\S]{0,30}(?:©|copyright|\(c\)|all\s+rights?\s+reserved)"
        r"[\s\S]{0,300}$",
        re.IGNORECASE,
    ),
    # "Confidential" / "proprietary" disclaimers
    re.compile(
        r"^[\s\S]{0,30}(?:confidential|proprietary|internal\s+use\s+only)"
        r"[\s\S]{0,300}$",
        re.IGNORECASE,
    ),
    # Page number only  ("Page 3", "- 12 -", "3 / 10", "Trang 5")
    re.compile(
        r"^\s*(?:page|trang|p\.?)?\s*\d{1,4}\s*(?:[/of|trên]\s*\d{1,4})?\s*$",
        re.IGNORECASE,
    ),
    # Repeated dashes / underscores / equals (visual separators)
    re.compile(r"^\s*[-_=~*]{4,}\s*$"),
    # "Table of Contents" / "Mục lục" standalone headings
    re.compile(
        r"^\s*(?:table\s+of\s+contents?|mục\s+lục|nội\s+dung)\s*$",
        re.IGNORECASE,
    ),
    # Draft / watermark text
    re.compile(
        r"^\s*(?:draft|bản\s+nháp|watermark|confidential)\s*$",
        re.IGNORECASE,
    ),
    # Header/footer patterns: "Company Name | Page X" or "Report Title — 2024"
    re.compile(
        r"^[A-ZÀ-Ỹa-zà-ỹ\s\-|·•]{3,60}\s*[|·•\-—]\s*(?:page|trang|p\.?)?\s*\d{0,4}\s*$",
        re.IGNORECASE,
    ),
]

# Vietnamese legal boilerplate fragments (partial match — if chunk CONTAINS these
# AND is short, it's likely boilerplate)
_LEGAL_FRAGMENTS_VI = [
    "theo quy định của pháp luật",
    "không được sao chép",
    "bảo mật thông tin",
    "điều khoản sử dụng",
    "chịu trách nhiệm trước pháp luật",
    "bản quyền thuộc về",
]

_LEGAL_FRAGMENTS_EN = [
    "all rights reserved",
    "without prior written consent",
    "this document is confidential",
    "for internal use only",
    "subject to change without notice",
    "disclaimer:",
    "terms and conditions",
    "no part of this publication",
]


def _normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _content_hash(text: str) -> str:
    """SHA-256 of normalized text."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Generate character-level n-gram shingles from normalized text."""
    normed = _normalize_text(text)
    if len(normed) < n:
        return {normed}
    return {normed[i : i + n] for i in range(len(normed) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ── Stage 1: Noise Filter ───────────────────────────────────────────────

def _is_boilerplate(text: str) -> bool:
    """Check if text matches known boilerplate patterns."""
    stripped = text.strip()

    # Full-match patterns
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(stripped):
            return True

    # Short chunks with legal fragments
    normed = stripped.lower()
    if len(stripped) < 300:
        for frag in _LEGAL_FRAGMENTS_VI + _LEGAL_FRAGMENTS_EN:
            if frag in normed:
                return True

    return False


def _meaningful_char_count(text: str) -> int:
    """Count non-whitespace, non-punctuation characters."""
    return len(re.sub(r"[\s\-_=~*|#>•·\"\'`(){}\[\]]+", "", text))


def filter_noise(chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
    """
    Stage 1: Remove chunks that are predominantly noise.

    Removes:
      - Chunks shorter than DEDUP_MIN_CHUNK_LENGTH meaningful characters
      - Boilerplate headers/footers/legal disclaimers/copyright notices
      - Whitespace-only or formatting-only chunks

    Preserves chunks with image_refs or table_refs regardless of text length,
    since their enriched captions carry semantic value.
    """
    min_len = settings.EXPLORERAG_DEDUP_MIN_CHUNK_LENGTH
    kept: list[EnrichedChunk] = []
    removed = 0

    for chunk in chunks:
        # Always keep chunks with attached images/tables
        if chunk.image_refs or chunk.table_refs:
            kept.append(chunk)
            continue

        text = chunk.content.strip()

        # Empty / whitespace-only
        if not text:
            removed += 1
            continue

        # Too short (after stripping formatting)
        if _meaningful_char_count(text) < min_len:
            removed += 1
            continue

        # Boilerplate match
        if _is_boilerplate(text):
            removed += 1
            continue

        kept.append(chunk)

    if removed:
        logger.info(f"Noise filter: removed {removed}/{len(chunks)} boilerplate/short chunks")

    return kept


# ── Stage 2: Exact Dedup ────────────────────────────────────────────────

def dedup_exact(chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
    """
    Stage 2: Remove chunks with identical normalized content.

    Uses SHA-256 of lowercased, whitespace-collapsed text. First occurrence wins.
    """
    seen_hashes: set[str] = set()
    kept: list[EnrichedChunk] = []
    removed = 0

    for chunk in chunks:
        h = _content_hash(chunk.content)
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        kept.append(chunk)

    if removed:
        logger.info(f"Exact dedup: removed {removed}/{len(chunks)} identical chunks")

    return kept


# ── Stage 3: Near-duplicate Detection ───────────────────────────────────

def dedup_near(
    chunks: list[EnrichedChunk],
    threshold: float | None = None,
) -> list[EnrichedChunk]:
    """
    Stage 3: Remove near-duplicate chunks using Jaccard similarity
    on character n-gram shingles.

    For each pair, the LATER chunk (by chunk_index) is dropped when
    similarity >= threshold.  O(n²) but n is typically < 200 chunks per
    document, so this is fast enough.
    """
    if threshold is None:
        threshold = settings.EXPLORERAG_DEDUP_NEAR_THRESHOLD

    if threshold >= 1.0:
        return chunks  # disabled

    # Pre-compute shingles
    shingles = [_char_ngrams(c.content) for c in chunks]

    drop_indices: set[int] = set()

    for i in range(len(chunks)):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(chunks)):
            if j in drop_indices:
                continue
            sim = _jaccard_similarity(shingles[i], shingles[j])
            if sim >= threshold:
                drop_indices.add(j)

    kept = [c for idx, c in enumerate(chunks) if idx not in drop_indices]
    removed = len(drop_indices)

    if removed:
        logger.info(
            f"Near dedup (threshold={threshold:.2f}): "
            f"removed {removed}/{len(chunks)} near-duplicate chunks"
        )

    return kept


# ── Public API ───────────────────────────────────────────────────────────

def deduplicate_chunks(
    chunks: list[EnrichedChunk],
) -> tuple[list[EnrichedChunk], dict[str, int]]:
    """
    Run the full 3-stage deduplication pipeline.

    Returns:
        (filtered_chunks, stats) where stats = {
            "input": total input chunks,
            "noise_removed": count removed by noise filter,
            "exact_removed": count removed by exact dedup,
            "near_removed": count removed by near dedup,
            "output": total output chunks,
        }
    """
    if not settings.EXPLORERAG_DEDUP_ENABLED:
        return chunks, {"input": len(chunks), "output": len(chunks),
                        "noise_removed": 0, "exact_removed": 0, "near_removed": 0}

    total_input = len(chunks)

    # Stage 1: Noise filter
    after_noise = filter_noise(chunks)
    noise_removed = total_input - len(after_noise)

    # Stage 2: Exact dedup
    after_exact = dedup_exact(after_noise)
    exact_removed = len(after_noise) - len(after_exact)

    # Stage 3: Near dedup
    after_near = dedup_near(after_exact)
    near_removed = len(after_exact) - len(after_near)

    # Re-index chunk_index to be contiguous
    for i, chunk in enumerate(after_near):
        chunk.chunk_index = i

    stats = {
        "input": total_input,
        "noise_removed": noise_removed,
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "output": len(after_near),
    }

    total_removed = total_input - len(after_near)
    if total_removed:
        logger.info(
            f"Dedup pipeline: {total_input} → {len(after_near)} chunks "
            f"(-{total_removed}: noise={noise_removed}, exact={exact_removed}, "
            f"near={near_removed})"
        )

    return after_near, stats
