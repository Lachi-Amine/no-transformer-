from pipeline.knowledge_graph import KnowledgeGraph


def _entry(id_, text, keywords=None, tags=None, domain="biology"):
    return {
        "id": id_,
        "domain": domain,
        "keywords": keywords or [],
        "tags": tags or [],
        "text": text,
    }


def test_terms_built_from_keywords():
    e = _entry("amylase-001", "Amylase breaks starch.", keywords=["amylase", "starch", "enzyme"])
    g = KnowledgeGraph([e])
    assert "amylase" in g.term_to_ids
    assert "starch" in g.term_to_ids
    assert "amylase-001" in g.term_to_ids["amylase"]


def test_terms_built_from_tags_and_id_slug():
    e = _entry("plate-tectonics-001", "Continents drift.", tags=["geology", "earth"])
    g = KnowledgeGraph([e])
    assert "geology" in g.term_to_ids
    # id slug 'plate-tectonics' contributes 'plate' and 'tectonics'
    assert "plate" in g.term_to_ids
    assert "tectonics" in g.term_to_ids


def test_short_terms_dropped():
    e = _entry("e-001", "Short terms.", keywords=["a", "ab", "abc", "abcd"])
    g = KnowledgeGraph([e])
    # min length is 4
    assert "a" not in g.term_to_ids
    assert "ab" not in g.term_to_ids
    assert "abc" not in g.term_to_ids
    assert "abcd" in g.term_to_ids


def test_linked_ids_finds_other_entry_via_text():
    e1 = _entry("amylase-001", "Amylase breaks starch into glucose.", keywords=["amylase"])
    e2 = _entry("glucose-001", "Glucose is a simple sugar.", keywords=["glucose", "sugar"])
    g = KnowledgeGraph([e1, e2])
    linked = g.linked_ids(e1)
    assert "glucose-001" in linked


def test_linked_ids_excludes_self():
    e = _entry("amylase-001", "Amylase contains the word amylase.", keywords=["amylase"])
    g = KnowledgeGraph([e])
    linked = g.linked_ids(e)
    assert "amylase-001" not in linked


def test_linked_ids_excludes_provided():
    e1 = _entry("amylase-001", "Amylase breaks starch into glucose.", keywords=["amylase"])
    e2 = _entry("glucose-001", "Glucose is sugar.", keywords=["glucose"])
    e3 = _entry("starch-001", "Starch is a polysaccharide.", keywords=["starch"])
    g = KnowledgeGraph([e1, e2, e3])
    linked = g.linked_ids(e1, exclude_ids={"glucose-001"})
    assert "glucose-001" not in linked
    assert "starch-001" in linked


def test_linked_entries_returns_dicts():
    e1 = _entry("amylase-001", "Amylase breaks starch into glucose.", keywords=["amylase"])
    e2 = _entry("glucose-001", "Glucose is sugar.", keywords=["glucose"])
    g = KnowledgeGraph([e1, e2])
    linked = g.linked_entries(e1)
    assert any(e.get("id") == "glucose-001" for e in linked)


def test_word_boundary_match_avoids_substring_false_positive():
    # 'ion' should not match inside 'connection'
    e1 = _entry("ion-001", "An ion is a charged particle.", keywords=["ion"])
    e2 = _entry("network-001", "A network is a connection.", keywords=["network"])
    g = KnowledgeGraph([e1, e2])
    # ion-001's text doesn't mention 'network' so no link
    assert "network-001" not in g.linked_ids(e1)
    # network-001's text contains 'connection' but NOT standalone 'ion'
    assert "ion-001" not in g.linked_ids(e2)


def test_works_with_real_empirical_yaml():
    from pathlib import Path
    import yaml
    folder = Path(__file__).resolve().parent.parent / "knowledge" / "empirical"
    entries = []
    for path in folder.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            entries.extend(data)
    g = KnowledgeGraph(entries)
    # Find amylase entry
    amylase = next((e for e in entries if e.get("id") == "amylase-001"), None)
    assert amylase is not None
    linked = g.linked_ids(amylase)
    # Should find at least one other entry — amylase mentions glucose, starch, etc.
    assert len(linked) >= 1


# --- M15: edges, neighbors, find_by_term, stats ---

def test_edges_built_at_init():
    e1 = _entry("amylase-001", "Amylase breaks starch into glucose.", keywords=["amylase"])
    e2 = _entry("glucose-001", "Glucose is sugar.", keywords=["glucose"])
    e3 = _entry("isolated-001", "Random text with no other-entry terms.", keywords=["isolated"])
    g = KnowledgeGraph([e1, e2, e3])
    assert "glucose-001" in g.edges["amylase-001"]
    assert g.edges["isolated-001"] == set()


def test_neighbors_returns_set():
    e1 = _entry("a-001", "Mentions term-b.", keywords=["a"])
    e2 = _entry("b-001", "Mentions term-a.", keywords=["term-b", "b"])
    g = KnowledgeGraph([e1, e2])
    assert g.neighbors("a-001") == {"b-001"} or g.neighbors("a-001") == set()


def test_find_by_term():
    e = _entry("amylase-001", "x", keywords=["amylase", "enzyme"])
    g = KnowledgeGraph([e])
    found = g.find_by_term("amylase")
    assert len(found) == 1
    assert found[0]["id"] == "amylase-001"
    assert g.find_by_term("nonexistent") == []


def test_find_by_term_via_id_slug():
    e = _entry("plate-tectonics-001", "x", keywords=["geology"])
    g = KnowledgeGraph([e])
    found = g.find_by_term("plate")
    assert len(found) == 1
    found = g.find_by_term("tectonics")
    assert len(found) == 1


def test_stats_shape():
    e1 = _entry("alpha-001", "Mentions beta.", keywords=["alpha", "beta"], domain="biology")
    e2 = _entry("beta-001", "Mentions alpha.", keywords=["alpha", "beta"], domain="biology")
    e3 = _entry("gamma-001", "Isolated text with no other terms.", keywords=["gamma"], domain="physics")
    g = KnowledgeGraph([e1, e2, e3])
    s = g.stats()
    assert s["nodes"] == 3
    assert s["edges"] >= 1
    assert "gamma-001" in s["isolated_ids"]
    assert s["by_domain"]["biology"]["nodes"] == 2
    assert s["by_domain"]["physics"]["nodes"] == 1
    assert all(isinstance(t, tuple) and len(t) == 2 for t in s["most_connected"])
