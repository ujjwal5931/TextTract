"""
OCR evaluation metrics: Character Error Rate (CER) and logging.
No generative correction—compare pipeline output to ground truth only.
"""

import logging
from typing import List, Optional

# jiwer for CER (standard edit-distance at character level)
try:
    from jiwer import process_characters, cer
except ImportError:
    cer = None
    process_characters = None


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate: (S + D + I) / N where N = len(reference).
    S=substitutions, D=deletions, I=insertions. Lower is better.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    ref = reference.strip()
    hyp = hypothesis.strip()
    if cer is not None and process_characters is not None:
        return cer(ref, hyp)
    # Fallback: Levenshtein-like ratio
    return _char_error_rate_fallback(ref, hyp)


def _char_error_rate_fallback(ref: str, hyp: str) -> float:
    """Simple character-level edit distance / len(ref)."""
    n = len(ref)
    if n == 0:
        return 0.0 if not hyp else 1.0
    # O(n*m) DP
    prev = list(range(n + 1))
    for i, ch in enumerate(hyp):
        curr = [i + 1]
        for j in range(1, n + 1):
            cost = 0 if ref[j - 1] == ch else 1
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1] / n


def aggregate_cer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Average CER over multiple (ref, hyp) pairs. Pads with empty if lengths differ."""
    if not references and not hypotheses:
        return 0.0
    n = max(len(references), len(hypotheses))
    refs = (references + [""] * n)[:n]
    hyps = (hypotheses + [""] * n)[:n]
    errs = [compute_cer(r, h) for r, h in zip(refs, hyps)]
    return sum(errs) / len(errs)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for pipeline and metrics."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_cer_sample(reference: str, hypothesis: str, logger: Optional[logging.Logger] = None) -> float:
    """Compute and log CER for one sample; return CER."""
    c = compute_cer(reference, hypothesis)
    log = logger or logging.getLogger("metrics")
    log.info("CER = %.4f | ref=%r | hyp=%r", c, reference[:80], hypothesis[:80])
    return c
