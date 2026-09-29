# -*- coding: utf-8 -*-
"""
test_app.py — pytest smoke tests for the unified CFTC AI Assistant + Agent app.

These tests verify:
  1. All modules import cleanly (no syntax/import errors).
  2. The AI agent formatting helpers work correctly.
  3. The AI agent produces a response (Groq) — requires GROQ_API_KEY + TAVILY_API_KEY.
  4. The RAG chat completion works (Cosmos DB + Azure OpenAI) — requires valid
     .env Azure credentials. Marked so it can be skipped if unavailable.
  5. The FastAPI backend endpoint responds.
"""

import os
import sys

import pytest

# ---------------------------------------------------------------------
# Ensure we can import modules from the project root.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------
# 1. Module import sanity
# ---------------------------------------------------------------------
def test_import_rag_core():
    import rag_core
    assert hasattr(rag_core, "chat_completion")
    assert hasattr(rag_core, "generate_embeddings")
    assert hasattr(rag_core, "convert_document")
    assert hasattr(rag_core, "translate_text")
    assert hasattr(rag_core, "cftc_container")
    assert hasattr(rag_core, "cache_container")


def test_import_ai_agent():
    import ai_agent
    assert hasattr(ai_agent, "get_response_from_ai_agent")
    assert hasattr(ai_agent, "format_tool_response")
    assert hasattr(ai_agent, "groq_llm")
    assert hasattr(ai_agent, "get_search_tool")


def test_import_backend():
    import backend
    assert hasattr(backend, "app")
    assert hasattr(backend, "ALLOWED_MODEL_NAMES")
    assert "llama-3.1-8b-instant" in backend.ALLOWED_MODEL_NAMES


def test_import_app():
    import app
    assert hasattr(app, "demo")


# ---------------------------------------------------------------------
# 2. AI agent formatting helpers
# ---------------------------------------------------------------------
def test_format_tool_response_dict():
    from ai_agent import format_tool_response

    payload = {
        "results": [
            {
                "title": "CFTC Enforcement",
                "url": "https://www.cftc.gov/enforcement",
                "content": "A snippet about enforcement actions.",
            }
        ]
    }
    out = format_tool_response(payload)
    assert "CFTC Enforcement" in out
    assert "https://www.cftc.gov/enforcement" in out
    assert "A snippet about enforcement actions." in out


def test_format_tool_response_list():
    from ai_agent import format_tool_response

    payload = [
        {
            "title": "Result One",
            "link": "https://example.com/1",
            "snippet": "This is result one.",
        }
    ]
    out = format_tool_response(payload)
    assert "Result One" in out
    assert "https://example.com/1" in out
    assert "This is result one." in out


def test_format_tool_response_empty():
    from ai_agent import format_tool_response

    assert format_tool_response({}) == "No search results found."
    assert format_tool_response([]) == "No search results found."


def test_extract_final_answer():
    from ai_agent import _extract_final_answer

    response = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    assert "Hi there!" in _extract_final_answer(response)


# ---------------------------------------------------------------------
# 3. AI agent live response (Groq) — skipped if no key
# ---------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live Groq test.",
)
def test_agent_live_response():
    from ai_agent import get_response_from_ai_agent

    response = get_response_from_ai_agent(
        llm_id="llama-3.1-8b-instant",
        query="What is the CFTC?",
        allow_search=False,
        system_prompt="You are a helpful assistant.",
        provider="groq",
    )
    messages = response.get("messages", [])
    assert messages, "Expected at least one message in the response"
    content = messages[-1].get("content", "")
    assert content and content.strip(), "Expected a non-empty response content"


# ---------------------------------------------------------------------
# 4. RAG live completion — skipped if no Azure Cosmos/OpenAI keys
# ---------------------------------------------------------------------
_REQUIRED_RAG_KEYS = [
    "cosmos_uri",
    "cosmos_key",
    "cosmos_database_name",
    "cosmos_container_name",
    "openai_endpoint",
    "openai_key",
    "openai_embeddings_deployment",
    "openai_completions_deployment",
]


def _rag_keys_present():
    try:
        from dotenv import dotenv_values

        cfg = dotenv_values(".env")
        return all(cfg.get(k) for k in _REQUIRED_RAG_KEYS)
    except Exception:
        return False


@pytest.mark.skipif(
    not _rag_keys_present(),
    reason="Azure RAG credentials not present; skipping live RAG test.",
)
def test_rag_live_completion():
    from rag_core import chat_completion, cftc_container, cache_container

    answer, cached = chat_completion(
        cache_container, cftc_container, "What is the CFTC?"
    )
    assert answer and str(answer).strip(), "Expected a non-empty RAG answer"


# ---------------------------------------------------------------------
# 5. FastAPI backend endpoint
# ---------------------------------------------------------------------
def test_backend_health():
    from backend import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_backend_models():
    from backend import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/models")
    assert resp.status_code == 200
    assert "llama-3.1-8b-instant" in resp.json()["allowed_models"]


# ---------------------------------------------------------------------
# 6. Sentiment analysis tests (local engine — no Azure required)
# ---------------------------------------------------------------------
def test_sentiment_import():
    from sentiment.sentiment_analysis import analyze_text
    assert callable(analyze_text)


def test_sentiment_text_positive():
    from sentiment.sentiment_analysis import analyze_text

    res = analyze_text(
        "The division is pleased to grant this favorable no-action relief "
        "providing confidence for firms going forward.",
        engine="local",
    )
    assert res["label"] == "positive"
    assert res["polarity"] > 0
    assert res["engine"] == "local"


def test_sentiment_text_negative():
    from sentiment.sentiment_analysis import analyze_text

    res = analyze_text(
        "This is a terrible decision that will cause serious harm to " "participants.",
        engine="local",
    )
    assert res["label"] == "negative"
    assert res["polarity"] < 0


def test_sentiment_aggregate_helpers():
    from sentiment.sentiment_analysis import (
        analyze_text,
        aggregate_counts,
        average_polarity,
        format_results_table,
    )

    pos = analyze_text("This is great and wonderful news.", engine="local")
    neg = analyze_text("This is awful and terrible.", engine="local")
    counts = aggregate_counts([pos, neg])
    assert counts["positive"] == 1
    assert counts["negative"] == 1
    assert counts["neutral"] == 0
    avg = average_polarity([pos, neg])
    assert isinstance(avg, float)
    table = format_results_table([pos, neg])
    assert "positive" in table
    assert "negative" in table


def test_sentiment_analyze_file_pdf():
    # A real PDF exists in source_documents; verify the extraction pipeline works.
    import os
    from sentiment.sentiment_analysis import analyze_file

    pdf_path = os.path.join("source_documents", "02126.pdf")
    if not os.path.exists(pdf_path):
        import pytest
        pytest.skip("02126.pdf not present; skipping PDF analysis test.")
    res = analyze_file(pdf_path, engine="local")
    assert res["source"] == "02126.pdf"
    assert res["method"] == "pdfminer"
    assert res["label"] in ("positive", "neutral", "negative")
    assert res["text_len"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
