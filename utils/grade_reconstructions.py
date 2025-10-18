# utils/grade_reconstructions.py
# Pure Python, no external dependencies
from pathlib import Path
import csv
# Minimal BLOSUM62 matrix (for amino acids only)
BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3,
    ('C', 'C'): 9,
    ('Q', 'Q'): 5, ('Q', 'E'): 2,
    ('E', 'E'): 5,
    ('G', 'G'): 6,
    ('H', 'H'): 8, ('H', 'Y'): 2,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'V'): 1,
    ('K', 'K'): 5,
    ('M', 'M'): 5,
    ('F', 'F'): 6, ('F', 'Y'): 3,
    ('P', 'P'): 7,
    ('S', 'S'): 4, ('S', 'T'): 1,
    ('T', 'T'): 5,
    ('W', 'W'): 11,
    ('Y', 'Y'): 7,
    ('V', 'V'): 4,
}

def blosum_score(a, b):
    """Return symmetric BLOSUM62 score for amino acids a,b."""
    if a == "-" or b == "-":
        return -4  # approximate gap penalty
    return BLOSUM62.get((a, b), BLOSUM62.get((b, a), -1))

def levenshtein(s1, s2):
    """Compute Levenshtein distance using pure Python."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def grade_pair(seq1, seq2):
    """Compare two protein sequences using identity, conservative substitutions, and edit similarity."""
    if not seq1 or not seq2:
        return {"identity": 0, "similarity": 0, "norm_align": 0, "lev_sim": 0, "final_score": 0}

    L = min(len(seq1), len(seq2))
    identical, similar = 0, 0
    align_score = 0
    max_self = 0

    for a, b in zip(seq1, seq2):
        s = blosum_score(a, b)
        align_score += s
        if a == b:
            identical += 1
        elif s > 0:
            similar += 1
    # self-alignment max score estimate
    for a in seq1[:L]:
        max_self += blosum_score(a, a)

    pct_id = identical / L
    pct_sim = (identical + similar) / L
    norm_align = align_score / max_self if max_self > 0 else 0

    lev_sim = 1 - levenshtein(seq1, seq2) / max(len(seq1), len(seq2))
    final_score = 0.5 * norm_align + 0.3 * pct_sim + 0.2 * lev_sim

    return {
        "identity": round(pct_id, 4),
        "similarity": round(pct_sim, 4),
        "norm_align": round(norm_align, 4),
        "lev_sim": round(lev_sim, 4),
        "final_score": round(final_score, 4),
    }

def mean_grade(pairs, csv_path: str = "reconstruction_report.csv"):
    """
    Compute mean reconstruction quality and save detailed results to CSV.
    Each element of `pairs` is (orig, recon).
    """
    scores = []
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(exist_ok=True, parents=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "orig_len", "recon_len", "identity", "similarity",
            "norm_align", "lev_sim", "final_score"
        ])
        writer.writeheader()

        for i, (orig, recon) in enumerate(pairs, 1):
            metrics = grade_pair(orig, recon)
            metrics.update({
                "index": i,
                "orig_len": len(orig),
                "recon_len": len(recon),
            })
            writer.writerow(metrics)
            scores.append(metrics["final_score"])

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"[INFO] Wrote per-sequence metrics to {csv_path}")
    print(f"[INFO] Mean reconstruction score: {mean_score:.4f}")
    return mean_score
