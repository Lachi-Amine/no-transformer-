import pytest

from pipeline.schemas import (
    DOMAINS,
    INTENTS,
    Classification,
    EpistemicVector,
    EvidenceRecord,
    FusedEvidence,
    Query,
)


def test_epistemic_vector_valid():
    v = EpistemicVector(g=0.5, y=0.3, r=0.2)
    assert v.g == 0.5
    assert v.y == 0.3
    assert v.r == 0.2
    assert v.as_weights() == {"green": 0.5, "yellow": 0.3, "red": 0.2}


def test_epistemic_vector_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        EpistemicVector(g=0.5, y=0.5, r=0.5)


def test_epistemic_vector_no_negatives():
    with pytest.raises(ValueError, match="out of"):
        EpistemicVector(g=-0.1, y=0.5, r=0.6)


def test_epistemic_vector_no_above_one():
    with pytest.raises(ValueError, match="out of"):
        EpistemicVector(g=1.5, y=-0.3, r=-0.2)


def test_epistemic_vector_floating_tolerance():
    EpistemicVector(g=0.333333, y=0.333333, r=0.333334)


def test_query_construction():
    q = Query(
        raw="What is X?",
        normalized="what is x",
        tokens=("what", "is", "x"),
        entities={"enzyme": "amylase"},
    )
    assert q.raw == "What is X?"
    assert q.tokens == ("what", "is", "x")
    assert q.entities == {"enzyme": "amylase"}


def test_classification_construction():
    c = Classification(
        domain="biology",
        intent="define",
        domain_probs={"biology": 0.9, "medicine": 0.1},
        intent_probs={"define": 0.7, "explain_process": 0.3},
    )
    assert c.domain == "biology"
    assert c.intent == "define"
    assert c.domain_probs["biology"] == 0.9


def test_evidence_record():
    r = EvidenceRecord(
        engine="yellow",
        claim="Some claim",
        support=("amylase-001", "starch-001"),
        score=0.75,
    )
    assert r.engine == "yellow"
    assert len(r.support) == 2
    assert r.score == 0.75


def test_fused_evidence():
    r = EvidenceRecord(engine="green", claim="X", support=(), score=1.0)
    fe = FusedEvidence(records=(r,), contradictions=((0, 0),))
    assert len(fe.records) == 1
    assert fe.contradictions == ((0, 0),)


def test_constants_complete():
    assert "biology" in DOMAINS
    assert "physics" in DOMAINS
    assert "general" in DOMAINS
    assert len(DOMAINS) == 8

    assert "define" in INTENTS
    assert "compute" in INTENTS
    assert len(INTENTS) == 7
