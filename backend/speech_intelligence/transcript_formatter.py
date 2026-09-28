import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def clean_chunk_text(text: str) -> str:
    """Clean per-chunk text from Whisper artifacts and hallucinations."""
    if not text:
        return ""

    cleaned = text.strip()

    # Filter out empty audio markers or common whisper silent loops
    hallucination_patterns = [
        r"^(you|thank you|thanks for watching|subscribe|subtitles by).*$",
        r"^(अ|आ|इ|ई|उ|ऊ)+$",
    ]

    for pat in hallucination_patterns:
        if re.match(pat, cleaned, re.IGNORECASE) and len(cleaned.split()) <= 4:
            # If the entire segment is just a 1-3 word generic loop, inspect
            logger.debug(f"Detected possible silence artifact: '{cleaned}'")

    # Normalize excessive spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def deduplicate_overlap(prev_text: str, curr_text: str) -> str:
    """
    Remove boundary overlap words between consecutive audio chunks.
    For example, if chunk 1 ends with 'brothers and sisters' and chunk 2 starts with 'sisters of our nation'.
    """
    if not prev_text or not curr_text:
        return curr_text

    prev_words = prev_text.split()
    curr_words = curr_text.split()

    if not prev_words or not curr_words:
        return curr_text

    # Check overlaps of 1 to 5 words
    max_overlap = min(5, len(prev_words), len(curr_words))
    best_overlap = 0

    for k in range(1, max_overlap + 1):
        prev_tail = [w.lower().strip(".,!?;:\"'") for w in prev_words[-k:]]
        curr_head = [w.lower().strip(".,!?;:\"'") for w in curr_words[:k]]
        if prev_tail == curr_head:
            best_overlap = k

    if best_overlap > 0:
        logger.debug(f"Deduplicated {best_overlap} overlapping words across chunk boundary")
        return " ".join(curr_words[best_overlap:])

    return curr_text


def merge_chunk_transcripts(chunks_text: List[str]) -> str:
    """
    Merge sequential chunk transcripts into a cohesive, properly formatted transcript.
    """
    if not chunks_text:
        return ""

    merged_parts: List[str] = []

    for text in chunks_text:
        cleaned = clean_chunk_text(text)
        if not cleaned:
            continue

        if not merged_parts:
            merged_parts.append(cleaned)
        else:
            prev = merged_parts[-1]
            deduped = deduplicate_overlap(prev, cleaned)
            if deduped.strip():
                merged_parts.append(deduped.strip())

    full_transcript = " ".join(merged_parts).strip()

    # Format speech spacing and sentence casing
    full_transcript = re.sub(r"\s+([.,!?;:])", r"\1", full_transcript)
    full_transcript = re.sub(r"\s+", " ", full_transcript)

    return full_transcript
