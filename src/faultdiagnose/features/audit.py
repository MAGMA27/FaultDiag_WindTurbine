"""Training-only feature redundancy audit utilities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_STAT_SUFFIXES = ("_deriv2", "_deriv", "_skew", "_kurt", "_mean", "_std", "_fft")
_STAT_PRIORITY = {
    "_mean": 0,
    "_std": 1,
    "_deriv": 2,
    "_deriv2": 3,
    "_fft": 4,
    "_skew": 5,
    "_kurt": 6,
}


def _feature_priority(name: str) -> tuple[int, int, str]:
    """Prefer average/mean features over noisier higher-order statistics."""
    suffix = next((item for item in _STAT_SUFFIXES if name.endswith(item)), "")
    source = name[: -len(suffix)] if suffix else name
    source_priority = 0 if source.endswith("_avg") else 1
    return source_priority, _STAT_PRIORITY.get(suffix, len(_STAT_PRIORITY)), name


def _column_digest(values: np.ndarray) -> str:
    """Return a stable digest for exact-duplicate candidate detection."""
    return hashlib.sha1(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def audit_feature_matrix(
    features: pd.DataFrame,
    correlation_threshold: float = 0.995,
    variance_epsilon: float = 1e-10,
    drop_near_constant: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Audit training-only features and retain one feature per correlation component.

    Near-constant features are retained by default: in anomaly detection, a signal
    that is constant during normal operation can be highly informative when it moves.
    """
    if not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold must be between 0 and 1")
    if features.empty:
        raise ValueError("features must not be empty")
    values = features.to_numpy(dtype=np.float64, copy=False)
    finite_fraction = np.isfinite(values).mean(axis=0)
    variance = np.nanvar(values, axis=0)
    names = list(features.columns)
    stats = pd.DataFrame(
        {"feature": names, "finite_fraction": finite_fraction, "variance": variance}
    )
    stats["drop_reason"] = ""
    stats["near_constant"] = stats["variance"] <= variance_epsilon
    stats.loc[stats["finite_fraction"] < 1.0, "drop_reason"] = "non_finite"
    if drop_near_constant:
        stats.loc[
            (stats["drop_reason"] == "") & stats["near_constant"], "drop_reason"
        ] = "near_constant"

    candidate_indices = np.flatnonzero(stats["drop_reason"].to_numpy() == "")
    digest_to_index: dict[str, int] = {}
    for index in candidate_indices:
        if not drop_near_constant and bool(stats.loc[index, "near_constant"]):
            continue
        digest = _column_digest(values[:, index])
        prior = digest_to_index.get(digest)
        if prior is not None and np.array_equal(values[:, index], values[:, prior]):
            stats.loc[index, "drop_reason"] = f"duplicate_of:{names[prior]}"
        else:
            digest_to_index[digest] = index

    usable = np.flatnonzero(stats["drop_reason"].to_numpy() == "")
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(values[:, usable], rowvar=False)
    correlations = np.nan_to_num(correlations, nan=0.0)
    parent = np.arange(len(usable))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs = np.argwhere(np.triu(np.abs(correlations) >= correlation_threshold, k=1))
    for left, right in pairs:
        union(int(left), int(right))
    groups: dict[int, list[int]] = {}
    for position, index in enumerate(usable):
        groups.setdefault(find(position), []).append(int(index))

    position_by_index = {int(index): position for position, index in enumerate(usable)}
    cluster_rows: list[dict[str, object]] = []
    selected: list[str] = []
    for cluster_id, indices in enumerate(groups.values(), start=1):
        representative = min(indices, key=lambda index: _feature_priority(names[index]))
        selected.append(names[representative])
        rep_position = position_by_index[representative]
        for index in indices:
            position = position_by_index[index]
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": names[index],
                    "representative": names[representative],
                    "is_representative": index == representative,
                    "correlation_to_representative": float(
                        correlations[position, rep_position]
                    ),
                }
            )
    stats["selected"] = stats["feature"].isin(selected)
    return stats, pd.DataFrame(cluster_rows), selected
