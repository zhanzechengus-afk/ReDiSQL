from collections import Counter
from statistics import NormalDist
import math
import numpy as np


def wilson_interval(successes, total, confidence=0.95):
    if total < 1 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("invalid binomial counts or confidence")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def audit_summary(records):
    indexed = {(str(row["dataset"]), str(row["id"])): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("audit records must have unique IDs")
    groups = {}
    for row in records:
        label = row["label"]
        if label not in {"correct", "incorrect", "undecided"}:
            raise ValueError("audit labels must be correct, incorrect, or undecided")
        groups.setdefault(str(row["dataset"]), Counter())[label] += 1
    result = {}
    for dataset, counts in groups.items():
        total = sum(counts.values())
        result[dataset] = {"count": total}
        for label in ("correct", "incorrect", "undecided"):
            result[dataset][label] = {"count": counts[label], "rate": counts[label] / total, "wilson_95_ci": wilson_interval(counts[label], total)}
    return result


def paired_summary(first, second, bootstrap_samples=10000, seed=42):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or second.shape != first.shape or len(first) < 2:
        raise ValueError("paired summaries need at least two matched runs")
    if not np.isfinite(first).all() or not np.isfinite(second).all() or bootstrap_samples < 1:
        raise ValueError("scores must be finite and bootstrap_samples positive")
    delta = second - first
    generator = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 1000):
        size = min(1000, bootstrap_samples - start)
        indices = generator.integers(0, len(delta), size=(size, len(delta)))
        means[start:start + size] = delta[indices].mean(1)
    return {"paired_runs": len(delta), "first_mean": float(first.mean()), "first_sd": float(first.std(ddof=1)), "second_mean": float(second.mean()), "second_sd": float(second.std(ddof=1)), "mean_delta": float(delta.mean()), "delta_sd": float(delta.std(ddof=1)), "paired_bootstrap_95_ci": np.percentile(means, [2.5, 97.5]).tolist(), "deltas": delta.tolist(), "bootstrap_samples": bootstrap_samples, "seed": seed}

