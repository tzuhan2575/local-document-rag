from pathlib import Path

from local_document_rag.settings import load_settings


ENVIRONMENT_NAMES = [
    "QDRANT_PATH",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
]


def clear_environment(monkeypatch):
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_load_settings_uses_safe_defaults(monkeypatch):
    clear_environment(monkeypatch)

    settings = load_settings(env_file=None)

    assert settings.qdrant_path == Path("vector_store/qdrant")
    assert settings.qdrant_collection == "documents"
    assert settings.embedding_model
    assert settings.openai_model
    assert settings.openai_enabled is False


def test_load_settings_reads_overrides_without_storing_secret(
    monkeypatch,
):
    clear_environment(monkeypatch)
    monkeypatch.setenv("QDRANT_PATH", "custom/vector-store")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom-documents")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("OPENAI_MODEL", "custom-llm")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-value")

    settings = load_settings(env_file=None)

    assert settings.qdrant_path == Path("custom/vector-store")
    assert settings.qdrant_collection == "custom-documents"
    assert settings.embedding_model == "custom-embedding"
    assert settings.openai_model == "custom-llm"
    assert settings.openai_enabled is True
    assert "secret-test-value" not in repr(settings)
