from slugify import slugify


def test_basic():
    assert slugify(" Hello, World! ") == "hello-world"


def test_collapses_and_trims():
    assert slugify("a__b  c!!") == "a-b-c"
