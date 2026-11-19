"""Projection from each head's native labels onto the shared ontology.

mapped = M.T @ p_raw over a sparse matrix, so coverage can fall below one when a
head has no logit for a class. The residual is kept, not renormalised away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from .errors import MMSAError

__all__ = ["Ontology", "OntologyError"]


class OntologyError(MMSAError):
    """A malformed or inconsistent ontology definition."""

    def __init__(self, detail: str) -> None:
        super().__init__("MMSA-E000", f"Ontology definition is invalid: {detail}", "Check config/ontology.yaml.")


class Ontology:
    """Maps native emotion vocabularies onto the canonical six-class space."""

    def __init__(
        self,
        canonical: Sequence[str],
        matrices: Mapping[str, Mapping[str, Mapping[str, float]]],
        unsupported: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.canonical: tuple[str, ...] = tuple(canonical)
        self._raw: dict[str, dict[str, dict[str, float]]] = {k: dict(v) for k, v in matrices.items()}
        self.unsupported: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (unsupported or {}).items()
        }
        self._matrices: dict[str, np.ndarray] = {}
        self._labels: dict[str, tuple[str, ...]] = {}
        self.validate()
        self._build()

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Ontology":
        """Load an ontology definition from YAML."""
        p = Path(path)
        if not p.exists():
            raise OntologyError(f"file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        try:
            canonical = data["canonical"]
            matrices = {k: data[k] for k in ("acoustic", "text") if k in data}
        except KeyError as exc:  # pragma: no cover - config typo path
            raise OntologyError(f"missing key {exc}") from exc
        return cls(canonical=canonical, matrices=matrices, unsupported=data.get("unsupported"))

    # -- validation ---------------------------------------------------------- #

    def validate(self) -> None:
        """Check every documented invariant. Raises :class:`OntologyError`."""
        if not self.canonical:
            raise OntologyError("canonical label list is empty")
        if len(set(self.canonical)) != len(self.canonical):
            raise OntologyError("canonical labels contain duplicates")

        for branch, rows in self._raw.items():
            if not rows:
                raise OntologyError(f"branch '{branch}' has no rows")
            for raw_label, mapping in rows.items():
                if not mapping:
                    raise OntologyError(f"{branch}.{raw_label} maps to nothing")
                total = 0.0
                for canonical_label, weight in mapping.items():
                    if canonical_label not in self.canonical:
                        raise OntologyError(
                            f"{branch}.{raw_label} references unknown class '{canonical_label}'"
                        )
                    if weight < 0:
                        raise OntologyError(f"{branch}.{raw_label}.{canonical_label} is negative")
                    total += float(weight)
                if total > 1.0 + 1e-9:
                    raise OntologyError(
                        f"{branch}.{raw_label} row sums to {total:.3f} (> 1.0); "
                        "a row may not claim more mass than the model supplied"
                    )

    def _build(self) -> None:
        for branch, rows in self._raw.items():
            labels = tuple(rows.keys())
            matrix = np.zeros((len(labels), len(self.canonical)), dtype=np.float64)
            for i, label in enumerate(labels):
                for j, canonical_label in enumerate(self.canonical):
                    matrix[i, j] = float(rows[label].get(canonical_label, 0.0))
            self._labels[branch] = labels
            self._matrices[branch] = matrix

    # -- queries ------------------------------------------------------------- #

    def native_labels(self, branch: str) -> tuple[str, ...]:
        """Native vocabulary of a branch, in the order the model emits logits."""
        self._require(branch)
        return self._labels[branch]

    def matrix(self, branch: str) -> np.ndarray:
        """The ``(n_native, n_canonical)`` projection matrix (a copy)."""
        self._require(branch)
        return self._matrices[branch].copy()

    def is_supported(self, branch: str, canonical_label: str) -> bool:
        """Whether the branch has any vocabulary that can reach a class."""
        self._require(branch)
        j = self.canonical.index(canonical_label)
        return bool(self._matrices[branch][:, j].sum() > 0.0)

    def coverage_of_target_space(self, branch: str) -> float:
        """Fraction of canonical classes the branch can express at all."""
        self._require(branch)
        reachable = (self._matrices[branch].sum(axis=0) > 0.0).sum()
        return float(reachable) / float(len(self.canonical))

    def map_probs(
        self,
        branch: str,
        raw_labels: Iterable[str],
        raw_probs: Iterable[float],
    ) -> tuple[dict[str, float], float]:
        """Project native probabilities onto the canonical labels and report coverage.
        """
        self._require(branch)
        labels = tuple(raw_labels)
        probs = np.asarray(tuple(raw_probs), dtype=np.float64)

        expected = self._labels[branch]
        if labels != expected:
            # Order matters: a silently reordered vector would corrupt every
            # downstream number, so this is a hard failure, not a warning.
            if set(labels) == set(expected):
                order = [labels.index(lbl) for lbl in expected]
                probs = probs[order]
            else:
                raise OntologyError(
                    f"branch '{branch}' emitted labels {labels}, expected {expected}"
                )

        if probs.shape[0] != len(expected):
            raise OntologyError(
                f"branch '{branch}' supplied {probs.shape[0]} probabilities for "
                f"{len(expected)} labels"
            )
        if not np.all(np.isfinite(probs)):
            raise OntologyError(f"branch '{branch}' supplied non-finite probabilities")

        total = float(probs.sum())
        if total <= 0.0:
            raise OntologyError(f"branch '{branch}' supplied an all-zero distribution")
        probs = probs / total  # the raw softmax should already sum to 1

        mapped_vec = self._matrices[branch].T @ probs
        coverage = float(mapped_vec.sum())
        mapped = {label: float(mapped_vec[j]) for j, label in enumerate(self.canonical)}
        return mapped, coverage

    @staticmethod
    def renormalise(mapped: Mapping[str, float]) -> dict[str, float]:
        """Turn a sub-normalised vector into a proper distribution.

        Used **only** for the conflict index, where both branches must be proper
        distributions before a divergence is meaningful, and by the chart layer
        when it needs comparable shapes. Never used to build the fused vector.
        """
        total = sum(mapped.values())
        if total <= 0.0:
            uniform = 1.0 / max(len(mapped), 1)
            return {k: uniform for k in mapped}
        return {k: v / total for k, v in mapped.items()}

    # -- internals ----------------------------------------------------------- #

    def _require(self, branch: str) -> None:
        if branch not in self._matrices:
            raise OntologyError(
                f"unknown branch '{branch}'; available: {sorted(self._matrices)}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"Ontology(canonical={list(self.canonical)}, "
            f"branches={{k: len(v) for k, v in self._labels.items()}})"
        )
