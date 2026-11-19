from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserPreferenceInput(BaseModel):
    favorite_genres: list[str] = Field(default_factory=list)
    favorite_movie_ids: list[str] = Field(default_factory=list)
    description: str | None = None

    @staticmethod
    def _normalize_genres(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            result.append(cleaned)
        return result

    @property
    def normalized_genres(self) -> list[str]:
        return self._normalize_genres(self.favorite_genres)

    @property
    def normalized_movie_ids(self) -> list[str]:
        return list(dict.fromkeys(str(v).strip() for v in self.favorite_movie_ids if str(v).strip()))

    @property
    def normalized_description(self) -> str | None:
        if self.description is None:
            return None
        description = self.description.strip()
        if not description:
            return None
        return description


class UserPreferenceOut(BaseModel):
    user_id: str
    favorite_genres: list[str]
    favorite_movie_ids: list[str]
    description: str | None = None
    created_at: datetime
    updated_at: datetime
