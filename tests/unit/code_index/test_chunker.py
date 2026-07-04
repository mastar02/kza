"""Tests del chunker AST del code-index."""

from src.code_index.chunker import CodeChunk, extract_chunks

SAMPLE = '''\
import os


def top_level(a: int) -> int:
    """Suma uno."""
    return a + 1


@property_like
def decorated():
    return 2


class Foo:
    """Docstring de Foo."""

    ATTR = 1

    def method_a(self):
        return self.ATTR

    async def method_b(self):
        return 2


class Empty:
    """Sin métodos."""

    X = 5
'''


def test_top_level_function_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    fn = next(c for c in chunks if c.name == "top_level")
    assert fn.kind == "function"
    assert fn.chunk_id == "src/foo.py::top_level"
    assert fn.start_line == 4
    assert fn.end_line == 6
    assert '"""Suma uno."""' in fn.text
    assert fn.text.startswith("def top_level")


def test_decorator_included_in_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    fn = next(c for c in chunks if c.name == "decorated")
    assert fn.text.startswith("@property_like")
    assert fn.start_line == 9


def test_methods_have_qualified_name():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    names = {c.name for c in chunks}
    assert "Foo.method_a" in names
    assert "Foo.method_b" in names
    mb = next(c for c in chunks if c.name == "Foo.method_b")
    assert mb.kind == "method"
    assert mb.text.startswith("async def method_b")


def test_class_chunk_is_header_without_method_bodies():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    cls = next(c for c in chunks if c.name == "Foo")
    assert cls.kind == "class"
    assert '"""Docstring de Foo."""' in cls.text
    assert "ATTR = 1" in cls.text
    assert "def method_a" not in cls.text


def test_class_without_methods_is_full_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    cls = next(c for c in chunks if c.name == "Empty")
    assert "X = 5" in cls.text


def test_syntax_error_returns_empty():
    assert extract_chunks("def broken(:\n  pass", "src/bad.py") == []


def test_empty_source_returns_empty():
    assert extract_chunks("", "src/empty.py") == []
