"""Runtime configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from local_document_rag.embedder import DEFAULT_MODEL_NAME
from local_document_rag.openai_llm import DEFAULT_OPENAI_MODEL


@dataclass(frozen=True)
class AppSettings:
    qdrant_path: Path
    qdrant_collection: str
    embedding_model: str
    openai_model: str
    openai_enabled: bool


def _non_empty_environment_value(
    name: str,
    default: str,
) -> str:
    value = os.getenv(name, default).strip()

    if not value:
        raise ValueError(f"{name} must not be empty")

    return value


def load_settings(
    env_file: str | Path | None = ".env",
) -> AppSettings:
    """Load public configuration without storing the API key itself."""

    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)

    return AppSettings(
        qdrant_path=Path(
            _non_empty_environment_value(
                "QDRANT_PATH",
                "vector_store/qdrant",
            )
        ),
        qdrant_collection=_non_empty_environment_value(
            "QDRANT_COLLECTION",
            "documents",
        ),
        embedding_model=_non_empty_environment_value(
            "EMBEDDING_MODEL",
            DEFAULT_MODEL_NAME,
        ),
        openai_model=_non_empty_environment_value(
            "OPENAI_MODEL",
            DEFAULT_OPENAI_MODEL,
        ),
        openai_enabled=bool(
            os.getenv("OPENAI_API_KEY", "").strip()
        ),
    )
