import pytest

from pipeline import feedback


def test_record_and_load_roundtrip(tmp_path):
    path = tmp_path / "fb.csv"
    feedback.record("what is amylase", 1.0, "Empirically: Amylase is...", 0.7, path=path)
    feedback.record("what is xyz", 0.0, "I do not have sufficient knowledge", 0.2, path=path)

    rows = feedback.load(path=path)
    assert len(rows) == 2
    assert rows[0]["query"] == "what is amylase"
    assert rows[0]["label"] == 1.0
    assert rows[0]["confidence_at_time"] == 0.7
    assert rows[1]["label"] == 0.0


def test_record_appends_not_overwrites(tmp_path):
    path = tmp_path / "fb.csv"
    feedback.record("q1", 1.0, "ans1", 0.5, path=path)
    feedback.record("q2", 0.0, "ans2", 0.5, path=path)
    feedback.record("q3", 0.5, "ans3", 0.5, path=path)
    rows = feedback.load(path=path)
    assert len(rows) == 3
    assert [r["query"] for r in rows] == ["q1", "q2", "q3"]


def test_record_writes_header_once(tmp_path):
    path = tmp_path / "fb.csv"
    feedback.record("q1", 1.0, "ans1", 0.5, path=path)
    feedback.record("q2", 0.0, "ans2", 0.5, path=path)
    text = path.read_text(encoding="utf-8")
    assert text.count("timestamp,query,label") == 1


def test_record_includes_excerpt(tmp_path):
    path = tmp_path / "fb.csv"
    long_rendered = "Amylase is an enzyme. " * 20
    feedback.record("what is amylase", 1.0, long_rendered, 0.5, path=path)
    rows = feedback.load(path=path)
    assert "Amylase is an enzyme" in rows[0]["rendered_excerpt"]
    assert len(rows[0]["rendered_excerpt"]) <= feedback.EXCERPT_LEN


def test_record_excerpt_takes_first_line_only(tmp_path):
    path = tmp_path / "fb.csv"
    multiline = "First line claim.\nSecond line should not appear."
    feedback.record("q", 1.0, multiline, 0.5, path=path)
    rows = feedback.load(path=path)
    assert "First line claim" in rows[0]["rendered_excerpt"]
    assert "Second line" not in rows[0]["rendered_excerpt"]


def test_record_rejects_out_of_range_label(tmp_path):
    path = tmp_path / "fb.csv"
    with pytest.raises(ValueError, match="label must be"):
        feedback.record("q", 1.5, "ans", 0.5, path=path)
    with pytest.raises(ValueError, match="label must be"):
        feedback.record("q", -0.1, "ans", 0.5, path=path)


def test_record_rejects_empty_query(tmp_path):
    path = tmp_path / "fb.csv"
    with pytest.raises(ValueError, match="empty query"):
        feedback.record("", 1.0, "ans", 0.5, path=path)
    with pytest.raises(ValueError, match="empty query"):
        feedback.record("   ", 1.0, "ans", 0.5, path=path)


def test_load_missing_file_returns_empty(tmp_path):
    path = tmp_path / "missing.csv"
    assert feedback.load(path=path) == []


def test_load_empty_file_returns_empty(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert feedback.load(path=path) == []


def test_load_skips_malformed_rows(tmp_path):
    path = tmp_path / "fb.csv"
    path.write_text(
        "timestamp,query,label,confidence_at_time,rendered_excerpt\n"
        "2026-01-01,q1,not_a_float,0.5,ex\n"
        "2026-01-01,q2,1.0,0.5,ex\n",
        encoding="utf-8",
    )
    rows = feedback.load(path=path)
    assert len(rows) == 1
    assert rows[0]["query"] == "q2"
