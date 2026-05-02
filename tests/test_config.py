import pytest

from pipeline import config


@pytest.fixture(autouse=True)
def _reset_cache():
    config.reset_cache()
    yield
    config.reset_cache()


def test_defaults_load_when_no_file(tmp_path):
    cfg = config.load(path=tmp_path / "missing.yaml")
    assert cfg["yellow"]["bm25_threshold"] == 1.0
    assert cfg["fusion"]["min_engine_weight"] == 0.05
    assert cfg["orchestrator"]["history_size"] == 3


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "yellow:\n"
        "  bm25_threshold: 2.5\n"
        "  max_passages: 5\n"
        "fusion:\n"
        "  min_engine_weight: 0.10\n",
        encoding="utf-8",
    )
    cfg = config.load(path=p)
    assert cfg["yellow"]["bm25_threshold"] == 2.5
    assert cfg["yellow"]["max_passages"] == 5
    assert cfg["yellow"]["bm25_normalizer"] == 10.0  # untouched default
    assert cfg["fusion"]["min_engine_weight"] == 0.10


def test_get_helper(tmp_path):
    cfg = config.load(path=tmp_path / "missing.yaml")
    assert config.get("yellow", "bm25_threshold") == 1.0
    assert config.get("yellow", "missing_key", "default") == "default"
    assert config.get("nonexistent_section", "anything", 42) == 42


def test_malformed_yaml_falls_back_to_defaults(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("this: is: not: valid: yaml: [", encoding="utf-8")
    cfg = config.load(path=p)
    assert cfg["yellow"]["bm25_threshold"] == 1.0


def test_caches_after_first_load(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("yellow:\n  bm25_threshold: 7.7\n", encoding="utf-8")
    first = config.load(path=p)
    p.write_text("yellow:\n  bm25_threshold: 9.9\n", encoding="utf-8")
    second = config.load(path=p)  # cached, should not reflect the change
    assert first["yellow"]["bm25_threshold"] == 7.7
    assert second["yellow"]["bm25_threshold"] == 7.7
    config.reset_cache()
    third = config.load(path=p)
    assert third["yellow"]["bm25_threshold"] == 9.9
