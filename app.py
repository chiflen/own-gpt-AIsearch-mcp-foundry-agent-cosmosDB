# -*- coding: utf-8 -*-
"""
app.py — Unified Gradio Application

Integrates:
  1. CFTC RAG Document Q&A (Cosmos DB + Azure OpenAI)
  2. AI Agent (Azure OpenAI + Cosmos DB retrieval)
  3. Sentiment Analysis (documents, images, text)

All in a single Gradio tabbed interface.
"""

import os
import sys
import time
import logging

# ---------------------------------------------------------------------
# Bypass any system/corporate proxy for localhost so Gradio can bind
# and pass its local reachability check. Also disable Gradio analytics.
# ---------------------------------------------------------------------
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

import gradio as gr

# ---------------------------------------------------------------------
# Import RAG core (Cosmos DB, Azure OpenAI, vector search, etc.)
# ---------------------------------------------------------------------
from rag_core import (
    cftc_container,
    cache_container,
    chat_completion,
    convert_document,
    translate_text,
    LANGUAGE_CHOICES,
    CONVERT_TARGETS,
    ingest_documents,
    SOURCE_DIRECTORY,
    openai_completions_deployment as _default_deployment,
)

# ---------------------------------------------------------------------
# Import AI agent backend
# ---------------------------------------------------------------------
from ai_agent import get_response_from_ai_agent

# ---------------------------------------------------------------------
# Import Sentiment Analysis engine
# ---------------------------------------------------------------------
from sentiment.sentiment_analysis import (
    analyze_text,
    analyze_file,
    analyze_source_documents,
    analyze_blob_storage,
    aggregate_counts,
    average_polarity,
    format_results_table,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================================
# 1. RAG TAB  (CFTC Document Q&A)
# ====================================================================


def rag_user(user_message, chat_history):
    """Handle a RAG Q&A message."""
    start_time = time.time()
    response_payload, cached = chat_completion(
        cache_container, cftc_container, user_message
    )
    elapsed_time = round((time.time() - start_time) * 1000, 2)
    details = f"\n (Time: {elapsed_time}ms)"
    if cached:
        details += " (Cached)"
    chat_history.append([user_message, response_payload + details])
    return "", chat_history


def upload_and_ingest(files):
    """Upload files and re-ingest documents."""
    if not files:
        return "No files selected. Please choose one or more PDF/TXT documents first."
    if isinstance(files, str):
        files = [files]
    source_dir = SOURCE_DIRECTORY if SOURCE_DIRECTORY else "source_documents"
    os.makedirs(source_dir, exist_ok=True)
    saved = []
    for src in files:
        try:
            dest = os.path.join(source_dir, os.path.basename(src))
            with open(src, 'rb') as fin, open(dest, 'wb') as fout:
                fout.write(fin.read())
            saved.append(os.path.basename(dest))
        except Exception as e:
            print(f"  [error] failed to save {src}: {e}")
    if not saved:
        return "Failed to save any uploaded document."
    count = ingest_documents()
    return (
        f"Uploaded {len(saved)} document(s): {', '.join(saved)}\n\n"
        f"Ingested {count} item(s) into Cosmos DB."
    )


def do_convert(file_path, target_fmt):
    """Convert a document to the requested format."""
    status, out_path = convert_document(file_path, target_fmt)
    return status, (out_path if out_path else None)


def do_translate(text, src, tgt):
    """Translate text."""
    return translate_text(text, src, tgt)


# ====================================================================
# 2. AI AGENT TAB  (Azure OpenAI + Cosmos DB retrieval)
# ====================================================================

ALLOWED_MODELS = [
    _default_deployment or "gpt-5-csl14",
    "gpt-4o-mini",
    "gpt-4o",
]


def agent_respond(message, history, model_name, system_prompt, allow_search):
    """Handle an AI Agent (Azure OpenAI + Cosmos retrieval) message."""
    try:
        response = get_response_from_ai_agent(
            llm_id=model_name,
            query=message,
            allow_search=allow_search,
            system_prompt=system_prompt,
            provider="azure",
        )
        # Extract content from response
        messages = response.get("messages", [])
        if messages:
            content = messages[-1].get("content", "No response generated.")
        else:
            content = "No response generated."
        history.append([message, content])
        return "", history
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        history.append([message, f"**Error:** {str(e)}"])
        return "", history


# ====================================================================
# 3. SENTIMENT ANALYSIS TAB
# ====================================================================

SENTIMENT_ENGINES = ["auto (Azure + local fallback)", "local (offline)", "azure-openai"]


def _engine_key(engine_label):
    """Map a UI dropdown label back to the engine identifier."""
    if engine_label == "local (offline)":
        return "local"
    if engine_label == "azure-openai":
        return "azure"
    return "auto"


def sentiment_analyze_text(text, engine_label):
    """Analyze a typed-in text snippet."""
    if not text or not text.strip():
        return "Please enter some text to analyze."
    engine = _engine_key(engine_label)
    try:
        result = analyze_text(text, engine=engine)
    except Exception as e:
        return f"**Error:** {e}"
    label = result.get("label", "?")
    pol = result.get("polarity", 0)
    conf = result.get("confidence", 0)
    eng = result.get("engine", "?")
    summary = result.get("summary", "")
    lines = [
        f"**Sentiment:** {label}",
        f"**Polarity:** {pol}",
        f"**Confidence:** {conf}",
        f"**Engine:** {eng}",
    ]
    if summary:
        lines.append(f"**Summary:** {summary}")
    return "\n\n".join(lines)


def sentiment_analyze_files(files, engine_label):
    """Analyze one or more uploaded documents / images."""
    if not files:
        return "No files selected. Please upload at least one document or image."
    if isinstance(files, str):
        files = [files]
    engine = _engine_key(engine_label)
    results = []
    errors = []
    for f in files:
        try:
            res = analyze_file(f, engine=engine)
            results.append(res)
        except Exception as e:
            errors.append(f"{os.path.basename(f)}: {e}")
    return _format_sentiment_results(results, errors)


def sentiment_analyze_sources(engine_label):
    """Analyze all supported documents in the local source_documents folder."""
    engine = _engine_key(engine_label)
    try:
        results = analyze_source_documents(engine=engine)
    except Exception as e:
        return f"**Error:** {e}"
    if not results:
        return "No supported documents found in `source_documents/`."
    return _format_sentiment_results(results, [])


def sentiment_analyze_blob(engine_label):
    """Analyze all source documents in Azure Blob Storage."""
    engine = _engine_key(engine_label)
    try:
        results = analyze_blob_storage(engine=engine)
    except Exception as e:
        return f"**Error:** {e}"
    if not results:
        return "No documents found in blob storage (or blob storage unavailable)."
    return _format_sentiment_results(results, [])


def _format_sentiment_results(results, errors):
    """Format a list of sentiment result dicts into a readable markdown report."""
    counts = aggregate_counts(results)
    avg = average_polarity(results)
    pos, neu, neg, err = (
        counts.get("positive", 0),
        counts.get("neutral", 0),
        counts.get("negative", 0),
        counts.get("error", 0),
    )

    # Build a simple ASCII bar chart for the distribution.
    total = max(1, pos + neu + neg + err)
    bar = 30

    def _bar(n):
        blocks = int(bar * n / total)
        return "█" * blocks + "░" * (bar - blocks)

    lines = [
        "## Aggregate Sentiment Report",
        "",
        f"- **Total analyzed:** {total}",
        f"- **Average polarity:** {avg}",
        "",
        "### Distribution",
        "",
        f"🟢 Positive ({pos})  {_bar(pos)}",
        f"⚪ Neutral  ({neu})  {_bar(neu)}",
        f"🔴 Negative ({neg})  {_bar(neg)}",
        f"⚠️ Errors   ({err})  {_bar(err)}",
        "",
        "### Per-document results",
        "",
    ]
    lines.append(format_results_table(results))
    if errors:
        lines.append("\n### Errors\n")
        lines.extend(f"- {e}" for e in errors)
    return "\n".join(lines)


# ====================================================================
# BUILD UNIFIED APP
# ====================================================================

with gr.Blocks(title="CFTC AI Assistant + Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CFTC AI Assistant & Agent")
    gr.Markdown("*Developer: Norman Fletcher*")

    with gr.Tab("📄 CFTC Document Q&A"):
        gr.Markdown("## CFTC AI Assistant — Document Q&A")
        gr.Markdown(
            "Ask questions about CFTC enforcement action documents stored in the database."
        )

        rag_chatbot = gr.Chatbot(label="CFTC AI Assistant", type="tuples")
        with gr.Row():
            rag_msg = gr.Textbox(
                label="Ask me about legal document content in the Database!", scale=4
            )
            rag_btn = gr.Button("Enter", scale=1, variant="primary")
        rag_clear = gr.Button("Clear")

        # Upload & Ingest
        gr.Markdown("### Upload & Ingest Documents")
        gr.Markdown(
            "Select PDF or TXT documents to add to the knowledge base. They will be "
            "saved to `source_documents/` and automatically ingested into Cosmos DB."
        )
        with gr.Row():
            upload_files = gr.File(
                label="Choose documents (.pdf / .txt)",
                file_count="multiple",
                file_types=[".pdf", ".txt"],
                type="filepath",
            )
            upload_btn = gr.Button("Upload & Ingest Documents")
        ingest_status = gr.Markdown("")

        # Document Converter
        gr.Markdown("### Document Converter")
        with gr.Row():
            convert_file = gr.File(
                label="Choose a document to convert",
                file_count="single",
                file_types=[
                    ".txt", ".md", ".json", ".html", ".xml",
                    ".yaml", ".yml", ".csv", ".rtf", ".dat",
                    ".eml", ".pdf", ".docx",
                ],
                type="filepath",
            )
            convert_target = gr.Dropdown(
                label="Convert to",
                choices=list(CONVERT_TARGETS.keys()),
                value=".md",
            )
            convert_btn = gr.Button("Convert Document")
        convert_status = gr.Markdown("")
        convert_output = gr.File(label="Download converted file", interactive=False)

        # Translator
        gr.Markdown("### Translator")
        with gr.Row():
            translate_src = gr.Dropdown(
                label="Source language", choices=LANGUAGE_CHOICES, value="auto"
            )
            translate_tgt = gr.Dropdown(
                label="Target language",
                choices=[c for c in LANGUAGE_CHOICES if c != "auto"],
                value="English",
            )
        translate_input = gr.Textbox(label="Text to translate", lines=3)
        translate_btn = gr.Button("Translate")
        translate_output = gr.Textbox(label="Translation", lines=3, interactive=False)

        # Wire up events
        rag_msg.submit(
            rag_user, [rag_msg, rag_chatbot], [rag_msg, rag_chatbot], queue=False
        )
        rag_btn.click(
            rag_user, [rag_msg, rag_chatbot], [rag_msg, rag_chatbot], queue=False
        )
        rag_clear.click(lambda: None, None, rag_chatbot, queue=False)

        upload_btn.click(upload_and_ingest, [upload_files], [ingest_status], queue=False)
        convert_btn.click(
            do_convert,
            [convert_file, convert_target],
            [convert_status, convert_output],
            queue=False,
        )
        translate_btn.click(
            do_translate,
            [translate_input, translate_src, translate_tgt],
            [translate_output],
            queue=False,
        )

    with gr.Tab("🤖 AI Agent (Doc Retrieval)"):
        gr.Markdown("## AI Chatbot Agent — Azure OpenAI + Cosmos DB Retrieval")
        gr.Markdown(
            "Interact with an agent powered by Azure OpenAI. Toggle document "
            "retrieval to search the CFTC knowledge base in Cosmos DB."
        )

        agent_system_prompt = gr.Textbox(
            label="System Prompt (define agent personality)",
            lines=3,
            placeholder="You are a helpful assistant. Answer concisely and accurately.",
            value="You are a helpful assistant. Answer concisely and accurately.",
        )

        with gr.Row():
            agent_model = gr.Dropdown(
                label="Model",
                choices=ALLOWED_MODELS,
                value=ALLOWED_MODELS[0],
            )
            agent_allow_search = gr.Checkbox(
                label="Allow Document Retrieval (Cosmos DB)", value=True
            )

        agent_chatbot = gr.Chatbot(label="Agent Chat", type="tuples")
        with gr.Row():
            agent_msg = gr.Textbox(label="Your message", scale=4)
            agent_btn = gr.Button("Send", scale=1, variant="primary")
        agent_clear = gr.Button("Clear")

        # Wire up events
        def on_agent_submit(message, history, model, prompt, search):
            return agent_respond(message, history, model, prompt, search)

        agent_msg.submit(
            on_agent_submit,
            [agent_msg, agent_chatbot, agent_model, agent_system_prompt, agent_allow_search],
            [agent_msg, agent_chatbot],
            queue=False,
        )
        agent_btn.click(
            on_agent_submit,
            [agent_msg, agent_chatbot, agent_model, agent_system_prompt, agent_allow_search],
            [agent_msg, agent_chatbot],
            queue=False,
        )
        agent_clear.click(lambda: None, None, agent_chatbot, queue=False)

    with gr.Tab("❤️ Sentiment Analysis"):
        gr.Markdown("## Sentiment Analysis — Documents, Images & Text")
        gr.Markdown(
            "Analyze the sentiment (positive / neutral / negative) of documents, "
            "images, and text using a local offline engine (VADER + TextBlob) or "
            "Azure OpenAI. PDFs and images are text-extracted automatically via "
            "Azure Document Intelligence."
        )

        sentiment_engine = gr.Dropdown(
            label="Analysis Engine",
            choices=SENTIMENT_ENGINES,
            value=SENTIMENT_ENGINES[0],
        )

        # --- Text input analysis ---
        gr.Markdown("### Analyze Text")
        sentiment_text = gr.Textbox(label="Text to analyze", lines=4)
        sentiment_text_btn = gr.Button("Analyze Text", variant="primary")
        sentiment_text_out = gr.Markdown("")

        # --- Uploaded files / images ---
        gr.Markdown("### Analyze Uploaded Files / Images")
        gr.Markdown(
            "Upload one or more documents (.pdf / .txt / .csv / .json) or images "
            "(.png / .jpg) to analyze their sentiment."
        )
        sentiment_files = gr.File(
            label="Choose documents or images",
            file_count="multiple",
            file_types=[
                ".pdf", ".txt", ".md", ".csv", ".json", ".html", ".xml",
                ".yaml", ".yml", ".rtf", ".dat", ".eml",
                ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
            ],
            type="filepath",
        )
        sentiment_files_btn = gr.Button("Analyze Uploaded Files", variant="primary")
        sentiment_files_out = gr.Markdown("")

        # --- Source documents (local) ---
        gr.Markdown("### Source Documents (local)")
        sentiment_sources_btn = gr.Button("Analyze source_documents/ folder")
        sentiment_sources_out = gr.Markdown("")

        # --- Blob storage ---
        gr.Markdown("### Azure Blob Storage")
        sentiment_blob_btn = gr.Button("Analyze Blob Storage documents")
        sentiment_blob_out = gr.Markdown("")

        # Wire up sentiment events
        sentiment_text_btn.click(
            sentiment_analyze_text,
            [sentiment_text, sentiment_engine],
            [sentiment_text_out],
            queue=False,
        )
        sentiment_files_btn.click(
            sentiment_analyze_files,
            [sentiment_files, sentiment_engine],
            [sentiment_files_out],
            queue=False,
        )
        sentiment_sources_btn.click(
            sentiment_analyze_sources,
            [sentiment_engine],
            [sentiment_sources_out],
            queue=False,
        )
        sentiment_blob_btn.click(
            sentiment_analyze_blob,
            [sentiment_engine],
            [sentiment_blob_out],
            queue=False,
        )


# ====================================================================
# SMOKE TEST MODE
# ====================================================================
def smoke_test(port=7861):
    """Run a non-interactive smoke test (pytest-compatible)."""
    import httpx

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        debug=False,
        share=False,
        prevent_thread_lock=True,
    )
    base = f"http://127.0.0.1:{port}"
    results = {}

    def wait_ready(timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(base + "/", timeout=5, verify=False)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    results['root'] = wait_ready()

    for path in ("/config", "/info"):
        try:
            r = httpx.get(base + path, timeout=5, verify=False)
            results[path] = r.status_code == 200
        except Exception:
            results[path] = False

    print("\n===== SMOKE TEST RESULTS =====")
    for k, v in results.items():
        print(f"  {k}: {v}")
    demo.close()
    ok = all(v is True for v in results.values())
    print(("PASS" if ok else "FAIL"), "SMOKE TEST")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test(port=7861)
    else:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            debug=False,
            share=False,
        )
