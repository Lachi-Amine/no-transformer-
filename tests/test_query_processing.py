import pytest

from pipeline import query_processing


def test_basic_tokenization():
    q = query_processing.process("What is amylase?")
    assert q.normalized == "what is amylase?"
    assert q.tokens == ("what", "is", "amylase")


def test_normalization_lowercases():
    q = query_processing.process("WHAT IS XYZ")
    assert q.normalized == "what is xyz"


def test_strips_whitespace():
    q = query_processing.process("   what is x   ")
    assert q.raw == "   what is x   "
    assert q.normalized == "what is x"


def test_unicode_normalization():
    q = query_processing.process("café éclair")
    assert "café" in q.normalized
    assert "éclair" in q.normalized


def test_empty_query_rejected():
    with pytest.raises(ValueError, match="empty query"):
        query_processing.process("")


def test_whitespace_only_rejected():
    with pytest.raises(ValueError, match="empty query"):
        query_processing.process("   \t  ")


def test_tokenizer_drops_numbers():
    q = query_processing.process("compute 5 kg of mass")
    assert "5" not in q.tokens
    assert "kg" in q.tokens
    assert "mass" in q.tokens


def test_entity_extraction_enzyme():
    q = query_processing.process("the enzyme amylase breaks starch")
    assert q.entities.get("enzyme") == "amylase"


def test_entity_extraction_force():
    q = query_processing.process("compute the force F on the object")
    assert q.entities.get("force") == "f"


def test_no_entities_when_absent():
    q = query_processing.process("what is photosynthesis")
    assert q.entities == {}
