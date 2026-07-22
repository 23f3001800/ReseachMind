"""Tests for vector store persistence and the RAG capability flag.

These cover the catalog and the availability gate, which are pure logic. The
embedding round-trip itself is exercised separately — it downloads a model and
does not belong in a fast unit suite.
"""

import json

import pytest

from core import vectorstore


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """Point the store's paths at a throwaway directory."""
    monkeypatch.setattr(vectorstore, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(vectorstore, "INDEX_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setattr(vectorstore, "CATALOG_PATH", str(tmp_path / "documents.json"))
    return tmp_path


class TestCatalog:
    def test_missing_catalog_reads_as_empty(self, temp_store):
        assert vectorstore.load_catalog() == {}

    def test_round_trip(self, temp_store):
        vectorstore.save_catalog({"paper.pdf": {"text_length": 900, "num_chunks": 4}})
        assert vectorstore.load_catalog()["paper.pdf"]["num_chunks"] == 4

    def test_corrupt_catalog_reads_as_empty(self, temp_store):
        """A truncated write must not take the whole documents endpoint down."""
        (temp_store / "documents.json").write_text("{not json", encoding="utf-8")
        assert vectorstore.load_catalog() == {}

    def test_catalog_is_written_as_readable_json(self, temp_store):
        vectorstore.save_catalog({"a.txt": {"text_length": 1, "num_chunks": 1}})
        parsed = json.loads((temp_store / "documents.json").read_text(encoding="utf-8"))
        assert "a.txt" in parsed


class TestAvailabilityGate:
    def test_reports_a_boolean(self):
        assert isinstance(vectorstore.dependencies_available(), bool)

    def test_gate_is_not_merely_import_success(self, monkeypatch):
        """The module imports fine without faiss; the gate must still say False.

        Guards the regression where /health advertised rag_available: true on a
        slim image and every upload then failed.
        """
        import builtins

        real_import = builtins.__import__

        def no_faiss(name, *args, **kwargs):
            if name == "faiss":
                raise ImportError("no faiss here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_faiss)
        assert vectorstore.dependencies_available() is False
