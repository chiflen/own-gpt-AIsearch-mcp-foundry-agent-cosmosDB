# TechWyns AI Assistant + Agent

**Developed by Norman Fletcher**

A unified, tabbed **Gradio** application that combines three powerful capabilities for
TechWyns legal-document work:

1. **📄 TechWyns RAG Document Q&A** — a Retrieval-Augmented Generation (RAG) chatbot over TechWyns
   legal documents (enforcement actions) stored in an **Azure Cosmos DB** vector store, using
   **Azure OpenAI** for embeddings + chat completions, with smart caching, document upload,
   conversion, and translation.
2. **🤖 AI Agent (Doc Retrieval)** — an agent backed by **Azure OpenAI** with a **Cosmos DB
   vector-search tool**, a customizable system prompt, and selectable deployment models.
3. **❤️ Sentiment Analysis** — analyze the sentiment (positive / neutral / negative) of text,
   uploaded documents, images, the local `source_documents/` folder, and **Azure Blob Storage**
   using a local offline engine (**VADER + TextBlob**) or **Azure OpenAI**, with automatic text
   extraction via **Azure Document Intelligence** for PDFs and images.

> **Note:** The
> current version uses **Azure OpenAI**, **Azure Cosmos DB**, **Azure Blob Storage**, and
> **Azure Document Intelligence** (cloud) for high-quality, grounded answers and document
> processing.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Component Breakdown](#component-breakdown)
4. [Architectural Design](#architectural-design)
5. [Class Diagram](#class-diagram)
6. [Data Flow](#data-flow)
7. [Sequence Diagrams](#sequence-diagrams)
8. [System Requirements](#system-requirements)
9. [Environment Setup](#environment-setup)
10. [Configuration (.env)](#configuration-env)
11. [How To Run](#how-to-run)
12. [How To Run Each Test](#how-to-run-each-test)
13. [Usage Guide](#usage-guide)
14. [Azure Blob → Cosmos Ingestion Pipeline](#azure-blob--cosmos-ingestion-pipeline)
15. [Project Structure](#project-structure)
16. [Production & Optimization Notes](#production--optimization-notes)
17. [Smoke Test](#smoke-test)
18. [Disclaimer](#disclaimer)

---

## Features

### 📄 RAG Document Q&A Tab
- **Chatbot** — Ask questions about legal documents and get concise, cited answers grounded
  in the retrieved document chunks.
- **Smart caching** — Repeated/rephrased questions are answered instantly from the Cosmos DB
  cache (no repeated LLM calls) using a high similarity threshold (`0.99`).
- **Upload & Ingest Documents** — Upload one or more PDF/TXT files directly from the UI; they
  are saved to `source_documents/` and automatically ingested (chunked, embedded, and pushed
  to Cosmos DB). Idempotent upsert-by-id prevents duplicates.
- **Enter button** — A dedicated "Enter" button next to the input box (in addition to the
  Enter key) for triggering the question.
- **Document Converter** — Convert uploaded documents between formats: `.txt`, `.md`, `.json`,
  `.html`, `.xml`, `.yaml`, `.csv`, `.rtf`, `.dat`, `.eml`.
- **Translator** — Translate text between 13 languages (English, French, Spanish, Italian,
  Portuguese, German, Dutch, Chinese, Japanese, Arabic, Russian, Korean) using the Azure
  OpenAI completion model, with auto-detection.
- **Clear button** — Reset the conversation.

### 🤖 AI Agent (Doc Retrieval) Tab
- **AI Chatbot** — Uses the **Azure OpenAI** LLM to generate responses to user queries based
  on a defined system prompt.
- **Document Retrieval (Cosmos DB)** — When enabled, the agent calls a `cosmos_vector_search`
  tool that retrieves relevant TechWyns document chunks from Cosmos DB (reusing the RAG core) and
  feeds them back for a grounded final answer.
- **Customizable System Prompt** — Define the behavior and personality of the AI agent.
- **Model Selection** — Choose between several Azure OpenAI deployments (default from `.env`,
  plus `gpt-4o-mini`, `gpt-4o`, and legacy Groq-style names kept for backward compatibility).
- **Formatted Responses** — Retrieval results include document titles, snippets, similarity
  scores, and clickable source (Blob) links.

### ❤️ Sentiment Analysis Tab
- **Analyze Text** — Type in a snippet and get a sentiment label (positive / neutral /
  negative), polarity score, confidence, and (for Azure) a summary + aspects.
- **Analyze Uploaded Files / Images** — Upload PDFs, text files, or images; PDFs are parsed
  with `pdfminer`, images are OCR'd via **Azure Document Intelligence**, and text is analyzed.
- **Analyze Source Documents (local)** — Batch-analyze every supported file in `source_documents/`.
- **Analyze Blob Storage** — Download and analyze every source document in **Azure Blob Storage**.
- **Engine selection** — `auto (Azure + local fallback)`, `local (offline)`, or `azure-openai`.
- **Aggregate report** — Generates distribution bar charts, average polarity, per-document
  results, and error lists.

---

## Tech Stack

| Layer             | Technology                                                        | Purpose                                                        |
|-------------------|-------------------------------------------------------------------|----------------------------------------------------------------|
| **UI**            | Gradio (Blocks API)                                               | Tabbed web interface for chat, agent, sentiment, convert, translate |
| **Backend**       | Python 3.10+ (FastAPI optional wrapper)                           | Core application logic + optional API endpoints                |
| **Vector DB**     | Azure Cosmos DB (NoSQL + vector search)                           | Stores document chunks + embeddings, enables similarity search, Q&A cache |
| **Embeddings**    | Azure OpenAI (`text-embedding-3-large`, configurable dimensions)  | Converts text into numeric vectors                             |
| **LLM / Chat**    | Azure OpenAI (chat completions)                                   | Generates answers, agent responses, translations, sentiment    |
| **PDF parsing**   | `pdfminer.six`                                                     | Extracts text from PDF documents                               |
| **OCR / Doc AI**  | Azure Document Intelligence (Form Recognizer), foundry            | Extracts text from images and scanned PDFs                     |
| **Blob Storage**  | `azure-storage-blob`                                              | Stores source documents; provides clickable source links       |
| **Sentiment**     | `vaderSentiment` + `textblob` (local), Azure OpenAI (cloud)       | Deterministic offline + higher-quality cloud sentiment         |
| **Config**        | `python-dotenv` (`dotenv_values`)                                | Reads secrets and settings from `.env`                         |
| **Retry**         | `tenacity`                                                        | Resilient retries for embedding/LLM calls (exponential backoff)|
| **Server**        | Gradio built-in server (port 7861) + optional FastAPI/uvicorn    | Serves the UI; optional REST API                               |

---

## Component Breakdown

### 1. Configuration & Secrets — `rag_core.py` (`config = dotenv_values()`)
Reads the `.env` file for all Azure credentials and settings:
- `cosmos_uri`, `cosmos_key`, `cosmos_database_name`, `cosmos_container_name`
- `cosmos_vector_property_name`, `cosmos_cache_database_name`, `cosmos_cache_container_name`
- `openai_endpoint`, `openai_key`, `openai_api_version`
- `openai_embeddings_deployment`, `openai_embeddings_dimensions`
- `openai_completions_deployment`
- `SOURCE_DIRECTORY_SOURCE` (default `source_documents`)
- Blob storage: `blob_storage_url`, `blob_storage_container_name`, `blob_folder_path`

### 2. Azure Cosmos DB Setup — `rag_core.py`
Two vector-enabled containers are created (lazily, on first use):
- **TechWyns container** — stores ingested document chunks with their vector embeddings.
- **Cache container** — stores chat history and cached Q&A pairs (with vectors) for fast repeat answers.
- Both use a **vector embedding policy** (cosine distance, `float32`) and a **vector index
  policy** (`quantizedFlat`). Partition key is `/id`.
- **Lazy initialization** — a `_LazyContainer` proxy defers all Cosmos network I/O until the
  first method call, so importing `rag_core` / `app.py` / `ai_agent.py` / `backend.py` performs
  no network I/O (robust to transient outages).

### 3. Embedding Generation — `generate_embeddings(text)`
- Calls the Azure OpenAI embeddings API with configurable dimensions.
- Uses `@retry` from tenacity (exponential backoff, up to 20 attempts).

### 4. Document Ingestion — `ingest_documents()`
- Scans `source_documents/` recursively for `.pdf` and `.txt` files.
- Extracts text (`pdfminer` for PDF, plain read for TXT).
- Splits text into **overlapping chunks** (`CHUNK_SIZE=1500`, `CHUNK_OVERLAP=200`).
- Embeds each chunk and **upserts** it into the TechWyns container (idempotent by UUID).
- Builds a Blob URL for each source so responses can include clickable citations.
- Runs automatically on startup and on demand via the UI.

### 5. Vector Search — `vector_search(container, vectors, ...)`
- Queries Cosmos DB with `VectorDistance(c.vector, @embedding)` to find the most similar chunks.
- Returns top matches with similarity scores, source metadata, and blob links.

### 6. Chat History — `get_chat_history(container, completions=3)`
- Pulls the most recent completions from the cache container to provide conversational context.

### 7. Completion Generation — `generate_completion(...)`
- Builds a system prompt telling the model to answer **only** from the provided document context,
  provide at least 3 cited answers where applicable, and include source links (`blob_url`).

### 8. Caching — `cache_response` / `get_cache`
- Stores the generated answer + its embeddings in the cache container.
- On a new question, checks the cache with a very high similarity threshold (`0.99`). Near-identical
  questions return the cached answer instantly (marked **(Cached)** in the UI).

### 9. Document Converter — `convert_document` / `_text_to_format`
- Converts text between formats using Python stdlib + installed libs (`yaml`, `csv`, `xml`,
  `html`, `email`). Produces a downloadable file.

### 10. Translator — `translate_text(...)`
- Uses the Azure OpenAI chat completions model with a translation system prompt.
- Supports auto-detection of the source language.

### 11. AI Agent — `ai_agent.py`
- `get_response_from_ai_agent(...)` implements a **native OpenAI function-calling** agent
  (no langchain/langgraph/groq/tavily dependency).
- When `allow_search=True`, the agent calls the `cosmos_vector_search` tool (reusing
  `rag_core.generate_embeddings` + `vector_search`) to retrieve grounded TechWyns context.
- Bounded tool-calling loop (max 4 iterations) to avoid infinite loops.

### 12. Sentiment Engine — `sentiment/sentiment_analysis.py`
- `LocalAnalyzer` — offline VADER + TextBlob, deterministic, no network/API cost.
- `AzureOpenAIAnalyzer` — structured JSON sentiment via Azure OpenAI, with graceful fallback
  to the local analyzer on error.
- Text extraction: PDFs via `pdfminer`, images via Azure Document Intelligence, text files
  read directly, Blob storage downloaded and analyzed.

### 13. FastAPI Backend — `backend.py` (optional)
- Exposes `GET /health`, `GET /models`, `POST /chat`.
- Reuses the same `get_response_from_ai_agent` for the agent.

### 14. Compatibility Patches — applied at import time in `rag_core.py`
- Fixes `gradio` / `gradio_client` / `starlette` version incompatibilities affecting config
  serialization and template rendering (avoids 500s on `/config` and root page TypeErrors).

---

## Architectural Design

- **Single-process app**: `app.py` runs the unified Gradio UI (3 tabs) and all backend logic
  in one Python process, served on **http://127.0.0.1:7861**.
- **Cloud services**: Azure OpenAI (embeddings + chat), Azure Cosmos DB (vector store + cache),
  Azure Blob Storage (source documents + citations), and Azure Document Intelligence (OCR) are
  external cloud services accessed via SDKs.
- **Lazy cloud connections**: All Cosmos clients/containers are created only on first use,
  making the app importable and startable even during transient network/DNS outages.
- **Idempotent ingestion**: Documents are upserted by generated UUID, so re-running the app or
  re-ingesting is safe (no duplicates).
- **Two container separation**: The TechWyns container holds the knowledge base; the cache
  container holds conversation history and cached answers — keeping concerns separate and
  improving cache-hit latency.
- **Resilience**: `tenacity` retries smooth over transient failures on embeddings and LLM calls.
- **Grounded + cited answers**: The system prompt instructs the model to answer only from
  retrieved context and to include source document links (Blob URLs).
- **Sentiment resilience**: The local analyzer never requires Azure keys; the Azure analyzer
  always falls back to local results on any failure.

---

## Class Diagram

```
+------------------------------------------+        +------------------------------------------+
|              GradioApp (app.py)           |        |            RAGCore (rag_core.py)          |
+------------------------------------------+        +------------------------------------------+
| - demo : gr.Blocks (3 tabs)               |        | - config : dict (from .env)               |
| - rag_user()                             |        | - openai_client : AzureOpenAI             |
| - upload_and_ingest()                    |  uses  | - TechWyns_container : _LazyContainer         |
| - do_convert()                           | -----> | - cache_container : _LazyContainer        |
| - do_translate()                         |        | - generate_embeddings(text)               |
| - agent_respond()                        |        | - ingest_documents()                      |
| - sentiment_analyze_*()                  |        | - vector_search() / chat_completion()     |
+------------------------------------------+        | - convert_document() / translate_text()  |
                                                  +------------------------------------------+
                                                               |
                                                               | uses
                                                  +------------+------------+
                                                  |                         |
                                          +-----------------+      +--------------------------+
                                          |  AIAgent        |      |  SentimentAnalysis        |
                                          | (ai_agent.py)   |      | (sentiment_analysis.py)  |
                                          +-----------------+      +--------------------------+
                                          | get_response_   |      | LocalAnalyzer            |
                                          |  from_ai_agent()|      | AzureOpenAIAnalyzer      |
                                          | _cosmos_vector_ |      | analyze_text/file/       |
                                          |  search()       |      |  source_documents/blob() |
                                          +-----------------+      +--------------------------+
                                                               |
                                                  +----------------------+
                                                  |  FastAPI (backend.py)|
                                                  +----------------------+
                                                  | GET /health, /models |
                                                  | POST /chat           |
                                                  +----------------------+
```

---

## Data Flow — RAG Q&A

```
 PDF/TXT files in source_documents/
        |
        |  ingested on startup (and on demand via Upload & Ingest)
        v
[extract_text] -> [split_text (chunk)] -> [generate_embeddings] -> [upsert to Cosmos DB (TechWyns container)]
        |
        v
User asks a question in the UI
        |
        v
[generate_embeddings(user question)]
        |
        +--> [get_cache: near-identical match?] --> YES --> return cached answer (fast, "Cached")
        |
        | NO
        v
[vector_search: find similar chunks in TechWyns container]
        |
        v
[get_chat_history: recent context from cache container]
        |
        v
[generate_completion: LLM answers from retrieved chunks + citations]
        |
        v
[cache_response: store answer + embeddings]
        |
        v
Return answer to UI (with elapsed-time + "Cached" hints)
```

## Data Flow — AI Agent (Doc Retrieval)

```
User message + system prompt + model + allow_search
        |
        v
[get_response_from_ai_agent()]
        |
        +--> [allow_search?] --NO--> [LLM direct answer]
        |
        | YES
        v
[LLM decides to call cosmos_vector_search tool]
        |
        v
[generate_embeddings(query)] -> [vector_search(TechWyns container)]
        |
        v
[Tool results fed back to LLM]
        |
        v
[Grounded final answer with source links]
```

## Data Flow — Sentiment Analysis

```
Input (text | file | source_documents/ | Blob storage)
        |
        v
[extract_text_from_path / analyze_image_bytes]
   PDF -> pdfminer | Image -> Document Intelligence OCR | Text formats -> read
        |
        v
[analyze_text(text, engine)]
   local -> VADER + TextBlob
   azure -> Azure OpenAI (structured JSON)
   auto  -> azure with local fallback
        |
        v
[aggregate_counts / average_polarity / format_results_table]
        |
        v
Aggregate sentiment report (distribution bars + per-document results)
```

---

## Sequence Diagrams

### RAG Q&A
```
User        Gradio UI        rag_core.py (app)       Azure OpenAI        Cosmos DB
 |              |                  |                       |                |
 |--question--->|                  |                       |                |
 |              |--submit--------->|                       |                |
 |              |                  |--embed(question)----->|                |
 |              |                  |<--vector-------------|                |
 |              |                  |--check cache------------------------>|
 |              |                  |<--(cached or miss)--------------------|
 |              |                  |--vector search----------------------->|
 |              |                  |<--top chunks--------------------------|
 |              |                  |--chat history------------------------>|
 |              |                  |<--history-----------------------------|
 |              |                  |--completion(question+chunks)--------->|
 |              |                  |<--answer-------------------------------|
 |              |                  |--cache answer------------------------>|
 |              |                  |--return answer--> |                    |
 |<--answer-----|                  |                   |                    |
```

### AI Agent (Doc Retrieval)
```
User        Gradio UI        ai_agent.py            Azure OpenAI        Cosmos DB
 |              |                  |                       |                |
 |--message---->|                  |                       |                |
 |              |--agent_respond-->|                       |                |
 |              |                  |--chat.completions(initial)----------->|
 |              |                  |<--tool_call (cosmos_vector_search)----|
 |              |                  |--embed(query)------------------------>|
 |              |                  |--vector_search------------------------>|
 |              |                  |<--chunks------------------------------|
 |              |                  |--chat.completions(grounded)---------->|
 |              |                  |<--final answer------------------------|
 |              |                  |--return messages--> |                  |
 |<--answer-----|                  |                     |                  |
```

---

## System Requirements

- **Python 3.10+** (project uses Python 3.11 in `.venv`).
- **Azure subscription** with:
  - An **Azure OpenAI** resource (embeddings + chat completions deployments).
  - An **Azure Cosmos DB** account with **vector search** enabled.
  - (Optional but recommended) **Azure Blob Storage** for source documents + citations.
  - (Optional) **Azure Document Intelligence** for OCR of images/scanned PDFs.
- **Internet access** (cloud services are used).
- ~2 GB+ free disk for the virtual environment and dependencies.

---

## Environment Setup

1. Create a virtual environment (recommended):
   ```shell
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```

> If your network uses a proxy that blocks `pip`, see the `scripts/` folder for the
> wheel-download workaround used during development.

---

## Configuration (.env)

Create a `.env` file in the project root with the following keys:

```ini
# Cosmos DB
cosmos_uri=<your-cosmos-uri>
cosmos_key=<your-cosmos-key>
cosmos_database_name=<database-name>
cosmos_container_name=<container-name>
cosmos_vector_property_name=vector
cosmos_cache_database_name=<cache-db-name>
cosmos_cache_container_name=<cache-container-name>

# Azure OpenAI
openai_endpoint=https://<resource>.openai.azure.com/
openai_key=<your-openai-key>
openai_api_version=2024-02-15-preview
openai_embeddings_deployment=text-embedding-3-large
openai_embeddings_dimensions=1536
openai_completions_deployment=gpt-4o-mini

# (Optional) Blob Storage — for source documents + clickable citations
blob_storage_url=https://<account>.blob.core.windows.net
blob_storage_container_name=<container>
blob_folder_path=https://<account>.blob.core.windows.net/<container>/twproj-source/
blob_storage_connection_string=<connection-string>

# (Optional) Azure Document Intelligence / Form Recognizer — for OCR
form_recognizer_endpoint=https://<resource>.cognitiveservices.azure.com/
form_recognizer_key=<key>
form_recognizer_model_id=prebuilt-document

# (Optional) Source directory
SOURCE_DIRECTORY_SOURCE=source_documents
```

> **Note:** `blob_storage_connection_string` is optional — if omitted, the app falls back to
> `DefaultAzureCredential`.

---

## How To Run

### Option 1 — Foreground (for development)
```shell
.venv\Scripts\python.exe app.py
```
This will:
1. Load `.env`.
2. Apply the Gradio/starlette compatibility patches.
3. Set up lazy Cosmos connections (created on first use).
4. Ingest all documents from `source_documents/` (if any).
5. Launch the unified Gradio UI at **http://127.0.0.1:7861** with three tabs:
   - **TechWyns Document Q&A** (RAG chatbot)
   - **AI Agent (Doc Retrieval)** (Azure OpenAI + Cosmos retrieval)
   - **Sentiment Analysis**

### Option 2 — Background via launcher (recommended for hosting)
```shell
.venv\Scripts\python.exe start_server.py
```
- Launches `app.py` in the background (no console window).
- Writes the process PID to `server.pid`.
- Writes output to `server.log` and `server.log.err`.
- Prints the URL: **http://127.0.0.1:7861**

### Verifying the server
```shell
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:7861/
# or use the included verification script
.venv\Scripts\python.exe scripts\verify_server.py
```

### Stopping the server
```shell
# Read the PID from server.pid and kill it
taskkill /PID <pid> /F
# or on PowerShell: Stop-Process -Id (Get-Content server.pid) -Force
```

---

## How To Run Each Test

Run all tests (from the project root):

```shell
.venv\Scripts\python.exe -m pytest test_app.py -v
```

### Test breakdown (`test_app.py`)
| Test | Purpose | Requires cloud? |
|------|---------|-----------------|
| `test_import_rag_core` | `rag_core` imports + key functions present | No |
| `test_import_ai_agent` | `ai_agent` imports + agent functions present | No |
| `test_import_backend` | `backend` imports + `ALLOWED_MODEL_NAMES` | No |
| `test_import_app` | `app` imports + `demo` Blocks object present | No |
| `test_format_tool_response_dict` | Agent tool-response formatting (dict) | No |
| `test_format_tool_response_list` | Agent tool-response formatting (list) | No |
| `test_format_tool_response_empty` | Agent tool-response formatting (empty) | No |
| `test_extract_final_answer` | Agent final-answer extraction | No |
| `test_agent_live_response` | Live agent response | **GROQ_API_KEY** (skipped if unset) |
| `test_rag_live_completion` | Live RAG chat completion | **Azure keys** (skipped if absent) |
| `test_backend_health` | FastAPI `/health` | No |
| `test_backend_models` | FastAPI `/models` | No |
| `test_sentiment_import` | Sentiment module imports | No |
| `test_sentiment_text_positive` | Local positive sentiment | No |
| `test_sentiment_text_negative` | Local negative sentiment | No |
| `test_sentiment_aggregate_helpers` | Aggregate counts / polarity / table | No |
| `test_sentiment_analyze_file_pdf` | PDF extraction + analysis | Needs `source_documents/02126.pdf` |

### Live / end-to-end smoke test
```shell
.venv\Scripts\python.exe app.py --smoke-test
```
This launches the app, checks `/`, `/config`, and `/info` for HTTP 200, prints `PASS`/`FAIL`,
and exits.

### Import/health checks
```shell
.venv\Scripts\python.exe scripts\check_imports.py
.venv\Scripts\python.exe scripts\verify_server.py
```

---

## Usage Guide

### RAG Q&A — Asking Questions
1. Type a question in the **"Ask me about legal document content in the Database!"** box.
2. Press **Enter** or click the **Enter** button.
3. The assistant answers using the most relevant document chunks. Repeat questions are answered
   instantly from cache (marked **Cached** with response time).

### RAG Q&A — Uploading & Ingesting Documents
1. In **Upload & Ingest Documents**, click **Choose documents** to select one or more PDF/TXT files.
2. Click **Upload & Ingest Documents**.
3. Files are saved to `source_documents/` and automatically ingested (chunked, embedded, pushed
   to Cosmos DB). A status message shows how many items were ingested.

### RAG Q&A — Converting Documents
1. In **Document Converter**, upload a document.
2. Choose the target format from the **Convert to** dropdown.
3. Click **Convert Document**.
4. A downloadable converted file appears below.

### RAG Q&A — Translating Text
1. In **Translator**, enter the text to translate.
2. Choose the **source language** (or `auto`) and the **target language**.
3. Click **Translate**.

### AI Agent
1. Set the **System Prompt** (agent personality).
2. Choose the **Model** deployment.
3. Toggle **Allow Document Retrieval (Cosmos DB)** to ground answers in the TechWyns knowledge base.
4. Type a message and click **Send** (or press Enter).

### Sentiment Analysis
1. Choose an **Analysis Engine** (`auto`, `local`, or `azure-openai`).
2. Analyze **Text**, **Uploaded Files/Images**, the **Source Documents** folder, or **Blob Storage**.
3. Review the aggregate sentiment report (distribution bars + per-document results).

---

## Azure Blob → Cosmos Ingestion Pipeline

`blobFormRecogniserCosmos.py` is an **Azure Functions (Python v2)** blob-triggered pipeline that
keeps the Cosmos knowledge base in sync with new/updated documents in blob storage:

- **Blob trigger** (`blob_trigger_ingest`) fires on `Microsoft.Storage.BlobCreated` events for
  the `twproj14/twproj-source` container.
- **Reads** the blob content, then calls **Azure Document Intelligence (Form Recognizer)** to
  extract text/content.
- **Upserts** a metadata record into Cosmos DB (stable, idempotent `blob-<hash>` id) including
  `id`, `title`, `source_path`, `blob_url`, `container`, `blob_name`, `chunk_text`, `timestamp`,
  `page_count`, `blob_size`, and `status`.
- **Backfill** (`sync_all_blobs`, queue-triggered) scans the whole source container and processes
  every existing blob — useful for documents uploaded before the trigger was enabled.
- **CLI entry point** (`python blobFormRecogniserCosmos.py`) runs the full backfill directly.

---

## Project Structure

```
Langgraph_Agent_TechWyns/
├── app.py                        # Unified Gradio application (3 tabs: RAG Q&A + AI Agent + Sentiment)
├── rag_core.py                   # Shared RAG backend (Cosmos, embeddings, ingest, cache, convert, translate)
├── ai_agent.py                   # Azure OpenAI agent with Cosmos vector-search tool (function-calling)
├── backend.py                    # Optional FastAPI backend (/chat, /health, /models)
├── start_server.py               # Background launcher (writes server.pid, server.log, server.log.err)
├── requirements.txt              # Python dependencies
├── test_app.py                   # pytest smoke tests (imports, agent, sentiment, backend)
├── sentiment/
│   ├── sentiment_analysis.py     # Sentiment engine (LocalAnalyzer + AzureOpenAIAnalyzer)
│   └── sentiment.py              # Legacy/standalone sentiment (not used by the app)
├── source_documents/             # Documents to ingest (PDF/TXT)
├── scripts/                      # Helper scripts (verify_server, check_imports, env helpers, etc.)
├── blobFormRecogniserCosmos.py   # Azure Functions blob->OCR->Cosmos ingestion pipeline
├── OCR-batch-process-docs/       # Batch OCR / document-processing helpers
├── rag-chatbot.py                # Legacy standalone RAG Gradio app (kept for reference)
├── .env                          # Azure secrets & configuration (not committed)
├── .venv/                        # Virtual environment (Python 3.11)
├── server.pid                    # PID of the running background server (generated)
├── server.log                    # Server stdout (generated)
├── server.log.err                # Server stderr (generated)
├── README.md                     # This documentation
└── TODO.md                       # Task/status tracking
```

---

## Production & Optimization Notes

- **Ingest pipeline**: Use the Azure Functions blob trigger for near-real-time ingestion on
  `BlobCreated` / `BlobUpdated` events (avoids polling; required on Flex Consumption plans).
- **Cosmos cost optimization**: Archive infrequently accessed data to Blob Storage Cool Tier;
  use lifecycle policies and versioning. Consider reducing Cosmos RU/s for cold containers.
- **Chunking**: Chunk text aggressively (300–500 tokens) to reduce embedding cost.
- **Store both OCR + original**: Keep the extracted text and the original document; you often
  need both later.
- **Secure access**: Use SAS tokens or Entra-based access for secure document retrieval.
- **Retry**: Wrap Cosmos/OpenAI calls with retry policies (especially on burst loads).
- **Semantic re-ranker**: Consider enabling a semantic re-ranker to refine query/search results
  (usage-based charges apply).
- **Frontend**: Beyond Gradio, SimpleChat or OpenWebUI could serve as alternative UIs.
- The original Azure configuration guide for each component/infrastructure is available at the
  TechWyns SharePoint configuration document (see the earlier project notes).

---

## Smoke Test

To verify the app end-to-end non-interactively (server starts, root page + config + info load):

```shell
.venv\Scripts\python.exe app.py --smoke-test
```

This launches the app, runs checks against `/`, `/config`, and `/info`, prints `PASS` or `FAIL`,
and exits.

---

## Disclaimer

This is a test/validation project to demonstrate a RAG chatbot over TechWyns legal documents, an
AI retrieval agent, and sentiment analysis. It is **not production-ready**. Answers and
sentiment determinations are generated by AI models and should be reviewed by a qualified
professional before any legal reliance.
