from pipeline import decomposition


def test_compare_verb():
    d = decomposition.detect("compare mitosis and meiosis")
    assert d is not None
    assert d.kind == "compare"
    assert d.parts == ("mitosis", "meiosis")


def test_compare_with_to():
    d = decomposition.detect("compare apples to oranges")
    assert d is not None
    assert d.parts == ("apples", "oranges")


def test_difference_between():
    d = decomposition.detect("what is the difference between dna and rna")
    assert d is not None
    assert d.parts == ("dna", "rna")


def test_difference_between_no_what():
    d = decomposition.detect("difference between cat and dog")
    assert d is not None
    assert d.parts == ("cat", "dog")


def test_vs():
    d = decomposition.detect("python vs java")
    assert d is not None
    assert d.parts == ("python", "java")


def test_versus():
    d = decomposition.detect("apples versus oranges")
    assert d is not None
    assert d.parts == ("apples", "oranges")


def test_trailing_question_mark_stripped():
    d = decomposition.detect("compare mitosis and meiosis?")
    assert d is not None
    assert d.parts == ("mitosis", "meiosis")


def test_no_match_for_what_is():
    assert decomposition.detect("what is amylase") is None


def test_no_match_for_empty():
    assert decomposition.detect("") is None
    assert decomposition.detect("   ") is None


def test_no_match_when_x_equals_y():
    assert decomposition.detect("compare x and x") is None


def test_multi_word_subjects():
    d = decomposition.detect("compare type 1 diabetes and type 2 diabetes")
    assert d is not None
    assert d.parts == ("type 1 diabetes", "type 2 diabetes")
