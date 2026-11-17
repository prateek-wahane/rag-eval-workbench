from ragwb.chunking import chunk_by_heading, chunk_fixed
from ragwb.config import CORPUS_DIR
from ragwb.evaluate import load_golden
from ragwb.generate import is_abstention, parse_citations
from ragwb.ingest import Document, load_corpus

DOC = Document("d", "Doc", "# Doc\n\n## A\none two three\n\n## B\nfour five")


def test_heading_chunks_carry_title_and_section():
    chunks = chunk_by_heading(DOC)
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[0].text.startswith("Doc > A\n")


def test_fixed_chunks_overlap_and_cover_all_words():
    doc = Document("d", "Doc", " ".join(str(i) for i in range(100)))
    chunks = chunk_fixed(doc, size=40, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].text.split()[-1] == "99"


def test_citation_parsing_and_abstention():
    assert parse_citations("Yes [2]. Also [1][2].") == [1, 2]
    assert is_abstention("I don't know based on the available documents.")
    assert not is_abstention("30 days [1].")


def test_golden_set_points_at_real_documents():
    doc_ids = {d.doc_id for d in load_corpus(CORPUS_DIR)}
    for item in load_golden():
        assert set(item.expected_docs) <= doc_ids, item.id
        assert (item.type == "unanswerable") == (not item.expected_docs), item.id
