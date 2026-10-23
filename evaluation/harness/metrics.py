"""
Operational evaluation metrics and deterministic answer/citation matchers.
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional


CANNED_ABSTENTION_SNIPPET = "i don't have sufficient verified evidence"

REFUSAL_PHRASES = [
    "i don't have sufficient",
    "does not contain",
    "cannot be answered",
    "not mentioned",
    "not found in the",
    "insufficient evidence",
    "no information",
    "not provided in the",
    "there is no mention",
    "cannot answer",
    "do not contain",
    "unanswerable",
    "not specified in the",
    "no details are provided"
]


def normalize_text(text: str) -> str:
    """Normalizes text for robust factual substring matching."""
    if not text:
        return ""
    # Lowercase, replace non-alphanumeric with spaces, collapse spaces
    text = text.lower()
    # Normalize common symbols
    text = text.replace("$", " $").replace("%", " % ").replace("°c", " deg c")
    # Collapse multiple whitespaces
    text = " ".join(text.split())
    return text


def is_abstained(answer: Optional[str]) -> bool:
    """
    Returns True if the system declined to answer, either via the canned message
    or by stating lack of evidence in the prompt response.
    """
    if not answer or not answer.strip():
        return True
    norm = normalize_text(answer)
    if CANNED_ABSTENTION_SNIPPET in norm:
        return True
    for phrase in REFUSAL_PHRASES:
        if phrase in norm:
            return True
    return False


def check_answer_correctness(answer: Optional[str], required_facts: List[str], forbidden_facts: List[str], is_answerable: bool) -> bool:
    """
    Evaluates answer correctness:
    - If unanswerable: correct iff system abstained.
    - If answerable: correct iff all required facts are present AND no forbidden fact is present AND system did not abstain.
    """
    if not answer:
        return not is_answerable

    abstained = is_abstained(answer)

    if not is_answerable:
        return abstained

    # Answerable question
    if abstained:
        return False  # Over-abstention

    norm_ans = normalize_text(answer)

    # Check that every required fact is present
    for req in required_facts:
        norm_req = normalize_text(req)
        # Check either exact substring or all words in requirement
        req_words = norm_req.split()
        if norm_req not in norm_ans and not all(w in norm_ans for w in req_words):
            return False

    # Check that forbidden facts are NOT asserted
    for forbid in forbidden_facts:
        norm_forbid = normalize_text(forbid)
        forbid_words = norm_forbid.split()
        if norm_forbid in norm_ans or (len(forbid_words) > 1 and all(w in norm_ans for w in forbid_words)):
            return False

    return True


def parse_citations(answer: Optional[str]) -> List[Tuple[str, int]]:
    """
    Extracts citations from answer text.
    Handles shapes like:
      [doc_name, chunk 3]
      [doc_name | chunk 3]
      [doc_name, chunk 3, page 1]
    """
    if not answer:
        return []
    citations = []
    # Pattern for bracketed citations containing 'chunk <num>'
    pattern = r"\[([^\]]+)\]"
    matches = re.findall(pattern, answer)
    for m in matches:
        chunk_m = re.search(r"chunk\s*(\d+)", m, re.IGNORECASE)
        if chunk_m:
            chunk_idx = int(chunk_m.group(1))
            # Document name is typically before comma or pipe
            parts = re.split(r"[,|]", m)
            doc_name = parts[0].strip()
            citations.append((doc_name, chunk_idx))
    return citations


def evaluate_citations(answer: Optional[str], trusted_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes citation validity and coverage.
    """
    if not answer or not answer.strip():
        return {"num_citations": 0, "valid_citations": 0, "citation_validity": 1.0, "citation_coverage": 0.0}

    citations = parse_citations(answer)
    trusted_refs = set()
    for c in trusted_chunks:
        doc = c.get("doc_name") or c.get("document") or ""
        cid = c.get("chunk_id") if c.get("chunk_id") is not None else c.get("chunk_index")
        trusted_refs.add((doc.lower(), cid))

    valid_count = 0
    for doc_name, cid in citations:
        # Match by doc base name and chunk id
        matched = False
        for t_doc, t_cid in trusted_refs:
            if (doc_name.lower() in t_doc or t_doc in doc_name.lower()) and cid == t_cid:
                matched = True
                break
        if matched:
            valid_count += 1

    validity = (valid_count / len(citations)) if citations else 1.0

    # Sentence coverage
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
    cited_sentences = 0
    for s in sentences:
        if re.search(r"\[[^\]]*chunk\s*\d+[^\]]*\]", s, re.IGNORECASE):
            cited_sentences += 1
    coverage = (cited_sentences / len(sentences)) if sentences else 0.0

    return {
        "num_citations": len(citations),
        "valid_citations": valid_count,
        "citation_validity": validity,
        "citation_coverage": coverage
    }


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for proportion k/n.
    Returns (point_estimate, ci_low, ci_high).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996  # 95%
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return p, max(0.0, lower_bound), min(1.0, upper_bound)


def compute_auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    """
    Computes ROC AUROC using Mann-Whitney U statistic.
    """
    if len(set(labels)) < 2:
        return None  # Cannot compute AUROC with only 1 class
    pairs = list(zip(scores, labels))
    # Sort descending by score
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Count concordant pairs
    correct_ranks = 0
    rank = len(pairs)
    # Simple Mann-Whitney
    rank_sum_pos = 0
    for idx, (score, label) in enumerate(sorted(pairs, key=lambda x: x[0])):
        if label == 1:
            rank_sum_pos += (idx + 1)
    u_val = rank_sum_pos - (n_pos * (n_pos + 1)) / 2
    return u_val / (n_pos * n_neg)
