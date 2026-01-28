"""Text utilities for test validation."""

import re
import unicodedata


def slugify_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for fuzzy comparison
    """
    # Normalize unicode characters and strip combining marks (accents, diacritics)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters, keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_similarity(expected: str, actual: str) -> float:
    """Calculate word-level similarity ratio between two texts.

    Args:
        expected: The ground truth text
        actual: The extracted/compared text

    Returns:
        Similarity ratio from 0.0 to 1.0, representing the fraction
        of expected words found in actual text
    """
    expected_words = set(slugify_text(expected).split())
    actual_words = set(slugify_text(actual).split())

    if not expected_words:
        return 1.0  # Empty expected text is always matched

    return len(expected_words & actual_words) / len(expected_words)
