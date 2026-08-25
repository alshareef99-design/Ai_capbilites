from app.rag.index import get_index
from app.rag.ingest import ingest_knowledge


def test_ingest_and_search_english():
    ingest_knowledge(reset=True)
    hits = get_index().search("three competitive quotes procurement", top_k=4)
    assert hits
    assert any("procurement" in (h["title"] + h["source"] + h["text"]).lower() for h in hits)


def test_search_arabic_leave():
    ingest_knowledge(reset=True)
    hits = get_index().search("الإجازة السنوية للموظف", top_k=4)
    assert hits
    joined = " ".join(h["text"] for h in hits)
    assert "21" in joined


def test_department_filter():
    ingest_knowledge(reset=True)
    hits = get_index().search("approval threshold", top_k=6, department="finance")
    assert hits
    assert all(h["department"] == "finance" for h in hits)


def test_replace_same_document_id():
    ingest_knowledge(reset=True)
    before = len(get_index().chunks)
    ingest_knowledge(reset=False)
    assert len(get_index().chunks) == before
