import pytest

from pipeline.orchestrator import Pipeline, _format_citation, render
from pipeline.schemas import (
    Classification,
    EpistemicVector,
    EvidenceRecord,
    FusedEvidence,
    Query,
)


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


# --- format_citation ---

def test_format_citation_single():
    assert _format_citation(("amylase-001",)) == "[source: amylase-001]"


def test_format_citation_multiple():
    out = _format_citation(("amylase-001", "starch-002"))
    assert out == "[sources: amylase-001, starch-002]"


def test_format_citation_filters_freeform():
    assert _format_citation(("freeform-sympy",)) == ""


def test_format_citation_filters_equations():
    assert _format_citation(("F = m*a",)) == ""


def test_format_citation_empty():
    assert _format_citation(()) == ""


# --- render ---

def _query():
    return Query(raw="x", normalized="x", tokens=("x",), entities={})


def _cls():
    return Classification(domain="biology", intent="define", domain_probs={}, intent_probs={})


def test_render_no_evidence_message():
    out = render(
        _query(), _cls(), EpistemicVector(g=0.34, y=0.33, r=0.33),
        FusedEvidence(records=()), 0.5,
    )
    assert "do not have sufficient knowledge" in out


def test_render_yellow_with_citation():
    rec = EvidenceRecord(
        engine="yellow", claim="Amylase breaks down starch.",
        support=("amylase-001",), score=0.8,
    )
    out = render(
        _query(), _cls(), EpistemicVector(g=0.2, y=0.7, r=0.1),
        FusedEvidence(records=(rec,)), 0.7,
    )
    assert "Empirically:" in out
    assert "[source: amylase-001]" in out


def test_render_green_yellow_red_order():
    g = EvidenceRecord(engine="green", claim="G", support=("g-1",), score=1.0)
    y = EvidenceRecord(engine="yellow", claim="Y", support=("y-1",), score=0.7)
    r = EvidenceRecord(engine="red", claim="R", support=("r-1",), score=0.6)
    # records intentionally out of order
    out = render(
        _query(), _cls(), EpistemicVector(g=0.4, y=0.3, r=0.3),
        FusedEvidence(records=(y, r, g)), 0.7,
    )
    g_pos = out.index("Formally:")
    y_pos = out.index("Empirically:")
    r_pos = out.index("Interpretively:")
    assert g_pos < y_pos < r_pos


def test_render_contradictions_block():
    a = EvidenceRecord(engine="yellow", claim="It is X.", support=("a-1",), score=0.7)
    b = EvidenceRecord(engine="yellow", claim="It is not X.", support=("b-1",), score=0.7)
    out = render(
        _query(), _cls(), EpistemicVector(g=0.2, y=0.7, r=0.1),
        FusedEvidence(records=(a, b), contradictions=((0, 1),)), 0.5,
    )
    assert "Sources disagree" in out
    assert "[yellow] It is X." in out
    assert "[yellow] It is not X." in out


# --- pipeline end-to-end ---

def test_pipeline_status_keys(pipeline):
    status = pipeline.status()
    assert set(status.keys()) == {
        "domain_classifier", "intent_classifier", "epistemic_router", "confidence",
    }
    for v in status.values():
        assert v in ("trained", "stub")


def test_pipeline_runs_amylase(pipeline):
    resp = pipeline.run("what is amylase")
    assert resp.query.raw == "what is amylase"
    assert resp.classification.domain in {"biology", "medicine"}
    assert abs(resp.epistemic.g + resp.epistemic.y + resp.epistemic.r - 1.0) < 1e-6
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.rendered  # non-empty


def test_pipeline_runs_compute(pipeline):
    resp = pipeline.run("compute the kinetic energy of a 5 kg object at 10 m/s")
    assert "Computed" in resp.rendered or "250" in resp.rendered


def test_pipeline_runs_freeform(pipeline):
    resp = pipeline.run("solve x^2 - 5*x + 6 = 0")
    assert "2" in resp.rendered and "3" in resp.rendered


def test_pipeline_engine_status_present(pipeline):
    resp = pipeline.run("what is photosynthesis")
    assert "engine_status" in resp.debug
    assert "yellow" in resp.debug["engine_status"]


def test_pipeline_empty_query_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline.run("")


# --- conversation memory (M9) ---

def _fresh():
    return Pipeline()


def test_history_grows_then_caps():
    p = _fresh()
    assert p.history == []
    p.run("what is amylase")
    p.run("what is photosynthesis")
    p.run("what is mitosis")
    assert len(p.history) == 3
    p.run("what is dna")
    assert len(p.history) == 3
    assert p.history[0].query.raw == "what is photosynthesis"
    assert p.history[-1].query.raw == "what is dna"


def test_forget_clears_history():
    p = _fresh()
    p.run("what is amylase")
    assert len(p.history) == 1
    p.forget()
    assert p.history == []


def test_coref_resolves_pronoun_to_previous_topic():
    p = _fresh()
    p.run("what is amylase")
    resp = p.run("how does it work")
    coref = resp.debug.get("coref")
    assert coref is not None
    assert coref["topic"] == "amylase"
    assert "amylase" in coref["expanded"]
    assert "amylase" in resp.query.tokens
    # Original raw is preserved for display
    assert resp.query.raw == "how does it work"


def test_coref_no_history_no_resolution():
    p = _fresh()
    resp = p.run("how does it work")
    assert resp.debug.get("coref") is None


def test_coref_no_pronoun_no_resolution():
    p = _fresh()
    p.run("what is amylase")
    resp = p.run("what is photosynthesis")
    assert resp.debug.get("coref") is None


def test_coref_topic_is_longest_distinctive_token():
    p = _fresh()
    p.run("what is photosynthesis")
    resp = p.run("explain it")
    assert resp.debug["coref"]["topic"] == "photosynthesis"


def test_coref_threading_across_three_turns():
    p = _fresh()
    p.run("what is amylase")
    resp2 = p.run("how does it work")
    # After turn 2, the topic of the LAST query is now amylase (resolved tokens include it)
    resp3 = p.run("where is it found")
    assert resp3.debug["coref"]["topic"] in {"amylase", "found", "where"}
    # Pronoun got expanded into something
    assert "it" not in resp3.debug["coref"]["expanded"].lower().split()


def test_reload_does_not_clear_history():
    p = _fresh()
    p.run("what is amylase")
    p.reload()
    assert len(p.history) == 1
