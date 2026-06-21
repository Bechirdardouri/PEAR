from pear.verifiers import (
    verify,
    verify_anls,
    verify_exact,
    verify_mc,
    verify_numeric,
)


def test_mc_basic():
    assert verify_mc("The answer is B.", "B")
    assert verify_mc("(C)", "C")
    assert verify_mc("Answer: d", "D")
    assert not verify_mc("I think it is A", "B")
    assert not verify_mc("no letter here", "A")


def test_numeric_tolerance():
    assert verify_numeric("about 42", "42")
    assert verify_numeric("the value is 41.5", "42")  # within 5%
    assert not verify_numeric("the value is 30", "42")
    assert verify_numeric("answer is -0.03", "0")     # zero with abs tol
    assert verify_numeric("3.5e2", "350")


def test_exact_normalization():
    assert verify_exact("Paris.", "paris")
    assert verify_exact("The capital is Paris", "paris")
    assert not verify_exact("Berlin", "paris")
    assert not verify_exact("anything", "")


def test_anls_pipe_refs():
    assert verify_anls("hello world", "hello world")
    assert verify_anls("helo world", "hello world")     # 1 edit
    assert verify_anls("totally wrong", "hello|hi|hey") is False
    # pipe-split refs: best-of
    assert verify_anls("hi", "hello|hi|hey")
    # list-of-refs: best-of
    assert verify_anls("hello", ["hello", "hi"])
    assert not verify_anls("totally wrong", ["hello", "hi"])


def test_dispatcher():
    assert verify("B", "B", "mc")
    assert verify("42", "42", "numeric")
    assert verify("paris", "Paris", "exact")
    assert verify("hello", "hello", "anls")
