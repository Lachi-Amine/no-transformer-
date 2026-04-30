import pytest

from engines.green_symbolic import (
    GreenSymbolicEngine,
    _extract_values,
    _freeform_compute,
    _looks_mathematical,
)
from engines.red_synthesis import RedSynthesisEngine
from engines.yellow_retrieval import YellowRetrievalEngine
from pipeline import query_processing
from pipeline.schemas import Classification


def _cls(domain="physics", intent="compute"):
    return Classification(domain=domain, intent=intent, domain_probs={}, intent_probs={})


@pytest.fixture(scope="module")
def green():
    return GreenSymbolicEngine()


@pytest.fixture(scope="module")
def yellow():
    return YellowRetrievalEngine()


@pytest.fixture(scope="module")
def red():
    return RedSynthesisEngine()


# --- GREEN ---

def test_green_loads_entries(green):
    assert len(green.entries) >= 10


def test_green_kinetic_energy_compute(green):
    q = query_processing.process("compute the kinetic energy of a 5 kg object at 10 m/s")
    res = green.run(q, _cls("physics", "compute"))
    assert res is not None
    assert "250" in res.claim
    assert res.score >= 0.9


def test_green_pendulum_compute_uses_g_constant(green):
    q = query_processing.process("compute the period of a pendulum of length 1 m")
    res = green.run(q, _cls("physics", "compute"))
    assert res is not None
    assert "T = 2.00" in res.claim or "T = 2.01" in res.claim


def test_green_define_intent_returns_formula(green):
    q = query_processing.process("what is the period of a pendulum")
    res = green.run(q, _cls("physics", "define"))
    assert res is not None
    assert "pendulum" in res.claim.lower()


def test_green_no_match_returns_none(green):
    q = query_processing.process("what is amylase")
    res = green.run(q, _cls("biology", "define"))
    assert res is None


def test_green_freeform_solve(green):
    q = query_processing.process("solve x^2 - 5*x + 6 = 0")
    res = green.run(q, _cls("math", "compute"))
    assert res is not None
    assert "2" in res.claim and "3" in res.claim


def test_green_freeform_integral(green):
    q = query_processing.process("compute the integral of x**2 from 0 to 2")
    res = green.run(q, _cls("math", "compute"))
    assert res is not None
    assert "8/3" in res.claim


def test_green_freeform_derivative(green):
    q = query_processing.process("derivative of x^3")
    res = green.run(q, _cls("math", "compute"))
    assert res is not None
    assert "3*x**2" in res.claim


def test_extract_values_units():
    vars_ke = {"E": "kinetic energy (J)", "m": "mass (kg)", "v": "speed (m/s)"}
    vals = _extract_values("compute the kinetic energy of a 5 kg object at 10 m/s", vars_ke)
    assert vals == {"m": 5.0, "v": 10.0}


def test_extract_values_descriptors():
    vars_n2 = {"F": "net force (N)", "m": "mass (kg)", "a": "acceleration (m/s^2)"}
    vals = _extract_values("compute force with mass 5 and acceleration 3", vars_n2)
    assert vals.get("m") == 5.0
    assert vals.get("a") == 3.0


def test_extract_values_explicit_assignments():
    vars_ohm = {"V": "voltage (V)", "I": "current (A)", "R": "resistance (ohm)"}
    vals = _extract_values("compute resistance with V = 12 and I = 2", vars_ohm)
    assert vals == {"V": 12.0, "I": 2.0}


def test_extract_values_synonym_velocity():
    vars_ke = {"E": "kinetic energy (J)", "m": "mass (kg)", "v": "speed (m/s)"}
    vals = _extract_values("compute kinetic energy with mass 5 and velocity 10", vars_ke)
    assert vals.get("m") == 5.0
    assert vals.get("v") == 10.0


def test_looks_mathematical():
    assert _looks_mathematical("solve x^2 - 5 = 0")
    assert _looks_mathematical("compute 7 factorial")
    assert _looks_mathematical("derivative of x^3")
    assert not _looks_mathematical("what is amylase")
    assert not _looks_mathematical("what is photosynthesis")


def test_freeform_compute_solve():
    res = _freeform_compute("solve 2*x + 4 = 10")
    assert res is not None
    assert "x = 3" in res


def test_freeform_compute_integral():
    res = _freeform_compute("compute the integral of cos(x) from 0 to pi")
    assert res is not None
    assert "0" in res


def test_freeform_compute_factorial_natural():
    res = _freeform_compute("compute 7 factorial")
    assert res is not None
    assert "5040" in res


# --- YELLOW ---

def test_yellow_loads_entries(yellow):
    assert len(yellow.entries) >= 20


def test_yellow_amylase(yellow):
    q = query_processing.process("what is amylase")
    res = yellow.run(q, _cls("biology", "define"))
    assert res is not None
    assert "amylase" in res.claim.lower()
    assert "amylase-001" in res.support


def test_yellow_filters_by_domain(yellow):
    q = query_processing.process("what is amylase")
    res = yellow.run(q, _cls("biology", "define"))
    assert res is not None
    for sid in res.support:
        entry = next((e for e in yellow.entries if e.get("id") == sid), None)
        assert entry is not None
        assert entry.get("domain") == "biology"


def test_yellow_no_match_for_off_topic(yellow):
    q = query_processing.process("the moon is bright")
    res = yellow.run(q, _cls("biology", "define"))
    assert res is None or "moon" not in res.claim.lower()


def test_yellow_general_domain_requires_keyword_overlap(yellow):
    q = query_processing.process("is lying ever justified")
    res = yellow.run(q, _cls("general", "interpret"))
    if res is not None:
        for sid in res.support:
            entry = next(e for e in yellow.entries if e.get("id") == sid)
            kws = {k.lower() for k in (entry.get("keywords") or [])}
            assert kws & {"lying", "lie", "deception", "honesty"}


# --- RED ---

def test_red_loads_entries(red):
    assert len(red.entries) >= 9


def test_red_trolley_problem(red):
    q = query_processing.process("should we pull the lever in the trolley problem")
    res = red.run(q, _cls("philosophy", "interpret"))
    assert res is not None
    claim_low = res.claim.lower()
    assert "utilitarian" in claim_low
    assert "deontological" in claim_low or "kantian" in claim_low
    assert "virtue" in claim_low


def test_red_no_match_returns_none(red):
    q = query_processing.process("what is the speed of light")
    res = red.run(q, _cls("physics", "define"))
    assert res is None
