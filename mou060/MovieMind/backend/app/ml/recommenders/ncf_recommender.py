"""NCF inference wrapper.

Translates raw MovieLens IDs into embedding indices and predicts ratings with
the trained Neural Collaborative Filtering model. This is the *only* place that
may call the model for a known MovieLens user; MovieMind account IDs must never
reach it (see ``ARTIFACTS.md`` §7).

Inference is always:
- On the configured device (CPU by default).
- Under ``torch.inference_mode()``.
- Batched, so callers can score many (user, movie) pairs efficiently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from app.ml.model_loader import LoadedModel


class UnknownMovieLensIdError(ValueError):
    """A raw MovieLens ID has no trained embedding index.

    ``kind`` is ``"user"`` or ``"movie"`` so the API layer can map to 404/422.
    """

    def __init__(self, kind: str, raw_id: object) -> None:
        self.kind = kind
        self.raw_id = raw_id
        super().__init__(f"Unknown MovieLens {kind} id: {raw_id}")


class NCFRecommender:
    """Predicts ratings for known MovieLens users/movies via the NCF model."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded
        self._model = loaded.model
        self._user_mapping = loaded.user_mapping
        self._movie_mapping = loaded.movie_mapping
        self._device = loaded.device
        # The model's linear head is unbounded, so clamp outputs to the trained
        # rating scale declared in the config (defaults match the plan: 0.5-5.0).
        self._rating_min = float(loaded.config.get("rating_min", 0.5))
        self._rating_max = float(loaded.config.get("rating_max", 5.0))

    @property
    def version(self) -> str:
        return self._loaded.version

    def known_user_ids(self) -> list[str]:
        return list(self._user_mapping.keys())

    def known_movie_ids(self) -> list[str]:
        return list(self._movie_mapping.keys())

    def has_user(self, raw_id: object) -> bool:
        return str(raw_id) in self._user_mapping

    def has_movie(self, raw_id: object) -> bool:
        return str(raw_id) in self._movie_mapping

    def _resolve_user(self, raw_id: object) -> int:
        idx = self._user_mapping.get(str(raw_id))
        if idx is None:
            raise UnknownMovieLensIdError("user", raw_id)
        return idx

    def _resolve_movie(self, raw_id: object) -> int:
        idx = self._movie_mapping.get(str(raw_id))
        if idx is None:
            raise UnknownMovieLensIdError("movie", raw_id)
        return idx

    def predict(self, user_id: object, movie_id: object) -> float:
        """Predicted rating for one (raw user id, raw movie id) pair."""
        return self.predict_batch([(user_id, movie_id)])[0]

    def predict_batch(self, pairs: Sequence[tuple[object, object]]) -> list[float]:
        """Predicted ratings for multiple (raw user id, raw movie id) pairs.

        Raises :class:`UnknownMovieLensIdError` on the first unknown ID;
        resolution happens up front so an unknown ID never causes an
        ``IndexError`` during the embedding lookup.
        """
        if not pairs:
            return []
        user_indices = [self._resolve_user(u) for u, _ in pairs]
        movie_indices = [self._resolve_movie(m) for _, m in pairs]

        user_t = torch.tensor(user_indices, dtype=torch.long, device=self._device)
        movie_t = torch.tensor(movie_indices, dtype=torch.long, device=self._device)

        with torch.inference_mode():
            scores = self._model(user_t, movie_t)

        raw = [float(v) for v in scores.detach().cpu().tolist()]
        return [min(max(value, self._rating_min), self._rating_max) for value in raw]

    def score_candidates(
        self, user_id: object, movie_ids: Iterable[object]
    ) -> list[tuple[object, float]]:
        """Score a list of candidate movies for one known user.

        Returns ``(raw_movie_id, predicted_rating)`` tuples. Unknown movie IDs
        raise (they should be filtered by the caller via :meth:`has_movie`).
        """
        movie_list = list(movie_ids)
        if not movie_list:
            return []
        scores = self.predict_batch([(user_id, m) for m in movie_list])
        return list(zip(movie_list, scores, strict=True))
