"""Unit tests for pure logic (no databases or API keys needed).

Run:  pytest tests/
"""
from config.models import DocumentChunk
from evaluation.benchmark import fact_recall
from reasoning.multihop import MultiHopReasoner

NAMES = ["Google", "Salesforce", "Hugging Face", "OpenAI", "AI"]


def test_names_in_text_finds_known_entities_case_insensitively():
    found = MultiHopReasoner._names_in_text("Which companies work with salesforce?", NAMES)
    assert found == ["Salesforce"]


def test_names_in_text_matches_multiword_names():
    found = MultiHopReasoner._names_in_text("How is Google linked to Hugging Face?", NAMES)
    assert set(found) == {"Google", "Hugging Face"}


def test_names_in_text_requires_whole_words():
    # "AI" must not match inside "said" / "maintain" / "OpenAI"
    assert MultiHopReasoner._names_in_text("He said they maintain it", NAMES) == []
    assert MultiHopReasoner._names_in_text("OpenAI shipped it", NAMES) == ["OpenAI"]


def test_names_in_text_returns_empty_when_nothing_matches():
    assert MultiHopReasoner._names_in_text("What is the weather today?", NAMES) == []


def test_fact_recall_counts_matching_facts():
    assert fact_recall("Google and OpenAI both work with Salesforce", ["Google", "OpenAI", "Cohere"]) == 2 / 3
    assert fact_recall("nothing relevant here", ["Google"]) == 0.0
    assert fact_recall("anything", []) == 0.0


def test_fact_recall_ignores_currency_symbols_and_does_not_match_inside_numbers():
    assert fact_recall("invested $20 billion in Nvidia", ["Nvidia", "20 billion"]) == 1.0
    # "2 billion" must not match inside "$20 billion"
    assert fact_recall("a $20 billion deal", ["2 billion"]) == 0.0
    assert fact_recall("acquired for $2 billion", ["2 billion"]) == 1.0


def test_document_chunk_supports_dict_style_access():
    chunk = DocumentChunk(
        chunk_id="a_0", text="hello", article_id="a",
        source="Wired", similarity=0.9, metadata={"title": "T"},
    )
    assert chunk.get("source") == "Wired"
    assert chunk["text"] == "hello"
    assert chunk.get("missing", "fallback") == "fallback"
    assert chunk.get("metadata", {}).get("title") == "T"
    assert chunk.chunk_index == 0  # optional field defaults safely
