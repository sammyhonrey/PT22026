"""Typed application settings.

Loaded from environment variables and an optional ``.env`` file. In
production the app must fail fast when required secrets are missing or
insecure — see ``Settings._enforce_production_safety``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]

_DEVELOPMENT_DEFAULT_SECRET = "dev-only-secret-change-me-not-for-production-0123456789"


class Settings(BaseSettings):
    """Central configuration for the MovieMind backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # --- Application identity -------------------------------------------------
    APP_NAME: str = "MovieMind API"
    APP_ENV: Environment = "development"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # --- Security -------------------------------------------------------------
    SECRET_KEY: str = _DEVELOPMENT_DEFAULT_SECRET
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, le=24 * 60)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, ge=1, le=365)
    REFRESH_COOKIE_SECURE: bool = False
    REFRESH_COOKIE_SAME_SITE: Literal["lax", "strict", "none"] = "lax"
    JWT_ALGORITHM: str = "HS256"

    # --- Database -------------------------------------------------------------
    # Example (Neon): postgresql+psycopg://user:password@host/dbname?sslmode=require
    DATABASE_URL: str = "postgresql+psycopg://moviemind:moviemind@localhost:5432/moviemind"

    # --- CORS -----------------------------------------------------------------
    # Comma-separated list of explicit frontend origins. Never use "*".
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- ML model -------------------------------------------------------------
    MODEL_ARTIFACT_DIR: str = "artifacts"
    MODEL_DEVICE: Literal["cpu", "cuda"] = "cpu"
    ENABLE_DEMO_ENDPOINTS: bool = True

    # --- Recommendations ------------------------------------------------------
    # Number of ratings a user needs before personalized recommendations replace
    # the popularity fallback. Once a user has enough ratings, a hybrid strategy
    # may use content-based and collaborative scores together when available.
    MIN_RATINGS_FOR_PERSONALIZATION: int = Field(default=5, ge=1)
    RECOMMENDATION_CONTENT_WEIGHT: float = Field(default=0.7, ge=0.0, le=1.0)
    RECOMMENDATION_COLLABORATIVE_WEIGHT: float = Field(default=0.3, ge=0.0, le=1.0)
    RECOMMENDATION_MIN_SIMILARITY: float = Field(default=0.05, ge=-1.0, le=1.0)

    # --- API hardening --------------------------------------------------------
    # In-process rate limits (requests per minute, per client IP).
    RATE_LIMIT_LOGIN: int = Field(default=10, ge=1)
    RATE_LIMIT_REGISTER: int = Field(default=5, ge=1)
    RATE_LIMIT_DEMO: int = Field(default=30, ge=1)
    # Maximum request body size accepted by the API, in bytes (default 1 MiB).
    MAX_REQUEST_BODY_BYTES: int = Field(default=1024 * 1024, ge=1024)
    # Comma-separated allowed Host header values; empty disables the check.
    ALLOWED_HOSTS: str = ""

    # --- Observability ----------------------------------------------------------
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        """Parsed CORS origin list."""
        if not self.CORS_ORIGINS.strip():
            return []
        return [o.strip().rstrip("/") for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_hosts(self) -> list[str]:
        """Parsed allowed-host list (empty = host check disabled)."""
        if not self.ALLOWED_HOSTS.strip():
            return []
        return [h.strip() for h in self.ALLOWED_HOSTS.split(",") if h.strip()]

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _no_wildcard_origins(cls, value: str) -> str:
        if "*" in value:
            raise ValueError("Wildcard CORS origins are not allowed; list explicit origins")
        return value

    @field_validator("API_V1_PREFIX")
    @classmethod
    def _prefix_slash_format(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("/"):
            raise ValueError("API_V1_PREFIX must start with '/'")
        return value.rstrip("/") or "/"

    @field_validator("REFRESH_COOKIE_SAME_SITE")
    @classmethod
    def _none_requires_secure_cookie(cls, value: str, info: ValidationInfo) -> str:
        if value == "none" and info.data.get("APP_ENV") != "production" and not info.data.get(
            "REFRESH_COOKIE_SECURE", False
        ):
            raise ValueError("SameSite=None requires REFRESH_COOKIE_SECURE=true outside production")
        return value

    def _enforce_production_safety(self) -> None:
        """Raise if production configuration is missing or insecure."""
        if self.APP_ENV != "production":
            return
        problems: list[str] = []
        if not self.SECRET_KEY or self.SECRET_KEY == _DEVELOPMENT_DEFAULT_SECRET:
            problems.append("SECRET_KEY is missing or still set to the development default")
        if self.DEBUG:
            problems.append("DEBUG must be false in production")
        if not self.DATABASE_URL:
            problems.append("DATABASE_URL is required in production")
        if problems:
            raise RuntimeError(
                "Refusing to start in production with unsafe configuration: " + "; ".join(problems)
            )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def refresh_cookie_secure(self) -> bool:
        """Production refresh cookies must always be Secure."""
        return self.is_production or self.REFRESH_COOKIE_SECURE

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance (safe to use as a FastAPI dependency)."""
    settings = Settings()
    settings._enforce_production_safety()
    return settings
