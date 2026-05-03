from pipeline import reasoning_rules
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.schemas import (
    Classification,
    EpistemicVector,
    EvidenceRecord,
    FusedEvidence,
    Query,
    Response,
)


def _entry(id_, text, keywords=None, domain="biology"):
    return {
        "id": id_,
        "domain": domain,
        "keywords": keywords or [],
        "tags": [],
        "text": text,
    }


def _resp(rendered, records=()):
    return Response(
        query=Query(raw="x", normalized="x", tokens=("x",), entities={}),
        classification=Classification(
            domain="biology", intent="define",
            domain_probs={}, intent_probs={},
        ),
        epistemic=EpistemicVector(g=0.34, y=0.33, r=0.33),
        evidence=FusedEvidence(records=tuple(records)),
        confidence=0.5,
        rendered=rendered,
        debug={},
    )


def _make_kg():
    return KnowledgeGraph([
        _entry("ph-001", "pH is the negative log of hydrogen ion concentration.",
               keywords=["ph", "acid", "hydrogen"], domain="chemistry"),
        _entry("henderson-hasselbalch-001",
               "Henderson-Hasselbalch: pH = pKa + log([A-]/[HA]).",
               keywords=["henderson", "hasselbalch", "pka", "buffer"],
               domain="chemistry"),
        _entry("kinetic-energy-001",
               "Kinetic energy of a moving object: E = (1/2) m v^2.",
               keywords=["kinetic", "energy", "mass", "velocity"], domain="physics"),
        _entry("newton-second-001",
               "Newton's second law: F = m a.",
               keywords=["newton", "force", "mass"], domain="physics"),
        _entry("catalyst-001",
               "A catalyst speeds up a chemical reaction by lowering activation energy.",
               keywords=["catalyst", "enzyme", "activation"], domain="chemistry"),
        _entry("compound-interest-001",
               "Compound interest grows balances geometrically.",
               keywords=["compound", "interest", "principal"], domain="economics"),
        _entry("arithmetic-mean-001",
               "Arithmetic mean: mu = sum / n.",
               keywords=["mean", "average", "arithmetic"], domain="math"),
    ])


def test_acid_rule_fires_on_acid_mention():
    kg = _make_kg()
    resp = _resp("Citric acid is found in lemons.")
    hits = reasoning_rules.apply_rules(resp, kg)
    rule_names = {h.rule_name for h in hits}
    assert "chemistry.acid_pH" in rule_names


def test_acid_rule_inhibited_when_ph_already_present():
    kg = _make_kg()
    resp = _resp("Citric acid has a pH of about 2.2.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "chemistry.acid_pH" not in {h.rule_name for h in hits}


def test_motion_rule_fires_on_motion_mention():
    kg = _make_kg()
    resp = _resp("Wind drives the motion of sand dunes.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "physics.motion_kinetic_energy" in {h.rule_name for h in hits}


def test_motion_rule_inhibited_when_kinetic_energy_present():
    kg = _make_kg()
    resp = _resp("The kinetic energy of the moving object is 250 J.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "physics.motion_kinetic_energy" not in {h.rule_name for h in hits}


def test_force_rule_fires():
    kg = _make_kg()
    resp = _resp("Gravity is a long-range force.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "physics.force_newton" in {h.rule_name for h in hits}


def test_enzyme_rule_fires():
    kg = _make_kg()
    resp = _resp("Amylase is an enzyme found in saliva.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "biology.enzyme_catalyst" in {h.rule_name for h in hits}


def test_interest_rule_fires():
    kg = _make_kg()
    resp = _resp("The bond pays 5 percent interest annually.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "economics.interest_compound" in {h.rule_name for h in hits}


def test_average_rule_fires():
    kg = _make_kg()
    resp = _resp("The average household income rose last year.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "math.average_mean" in {h.rule_name for h in hits}


def test_no_double_citation():
    kg = _make_kg()
    # Response already cites ph-001 — acid rule should not fire
    rec = EvidenceRecord(engine="yellow", claim="Acids have a pH below 7.",
                         support=("ph-001",), score=0.7)
    resp = _resp("Acids have a pH below 7.", records=(rec,))
    hits = reasoning_rules.apply_rules(resp, kg)
    assert "chemistry.acid_pH" not in {h.rule_name for h in hits}


def test_rule_record_has_see_also_prefix():
    kg = _make_kg()
    resp = _resp("Wind drives motion.")
    hits = reasoning_rules.apply_rules(resp, kg)
    motion_hits = [h for h in hits if h.rule_name == "physics.motion_kinetic_energy"]
    assert motion_hits
    assert motion_hits[0].record.claim.startswith("See also:")
    assert motion_hits[0].record.engine == "rule"


def test_no_rules_when_target_missing_from_kg():
    kg = KnowledgeGraph([])  # no entries
    resp = _resp("Wind drives motion.")
    hits = reasoning_rules.apply_rules(resp, kg)
    assert hits == []
