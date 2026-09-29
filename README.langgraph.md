# CFTCGPT — AI-Powered Legal Document Assistant + Langgraph Agent

**Developed by Norman Fletcher**

The CFTC AI Assistant is a unified, tabbed Gradio application that combines two powerful
capabilities:

1. **CFTC RAG Document Q&A** — a Retrieval-Augmented Generation (RAG) chatbot over CFTC legal
   documents (enforcement actions) stored in an Azure Cosmos DB vector store, using **Azure
   OpenAI** for embeddings + chat completions.
2. **AI Agent (Web Search)** — a Langgraph agent powered by the **Groq LLM** with **Tavily**
   web search, a customizable system prompt, and model selection.

> **Note:** This is a modern rewrite of the original local/private Streamlit prototype. The
> current version uses **Azure OpenAI**, **Azure Cosmos DB**, **Groq**, and **Tavily** (cloud)
> instead of fully-local models, which gives much better answer quality and good performance.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Sequence Diagram](#sequence-diagram)
6. [Architecture Design](#architecture-design)
7. [System Requirements](#system-requirements)
8. [Environment Setup](#environment-setup)
9. [Configuration (.env)](#configuration-env)
10. [How To Run](#how-to-run)
11. [Usage Guide](#usage-guide)
12. [Smoke Test](#smoke-test)
13. [Project Structure](#project-structure)
14. [Disclaimer](#disclaimer)

---

## Features

### RAG Document Q&A Tab
- **Chatbot** — Ask questions about legal documents and get concise, cited answers.
- **Smart caching** — Repeated questions are answered instantly from the Cosmos DB cache
  (no repeated LLM calls).
- **Upload & Ingest Documents** — Upload PDF/TXT files directly from the UI; they are saved
  to `source_documents/` and automatically ingested (chunked, embedded, and pushed to Cosmos DB).
- **Enter button** — A dedicated "Enter" button next to the input box (in addition to the
  Enter key) for triggering the question.
- **Document Converter** — Convert uploaded documents between formats: `.txt`, `.md`, `.json`,
  `.html`, `.xml`, `.yaml`, `.csv`, `.rtf`, `.dat`, `.eml`.
- **Translator** — Translate text between languages (English, French, Spanish, Italian,
  Portuguese, German, Dutch, Chinese, Japanese, Arabic, Russian, Korean) using the AI model.
- **Clear button** — Reset the conversation.

### AI Agent (Web Search) Tab
- **AI Chatbot** — Utilizes the Groq LLM to generate responses to user queries based on a
  defined system prompt.
- **Web Search Integration** — Incorporates the Tavily search tool to fetch relevant web
  search results and present them in a user-friendly format.
- **Customizable System Prompt** — Allows users to define the behavior and personality of the
  AI agent through a customizable system prompt.
- **Model Selection** — Choose between several Groq models (e.g.,gpt-5-csl14', `llama-3.1-8b-instant`,
  `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`, `llama3-70b-8192`).
- **Formatted Responses** — Formats web search results to include titles, snippets, and links,
  making the information easy to read and navigate.

---

## Tech Stack

| Layer          | Technology                                   | Purpose                                                          |
|----------------|----------------------------------------------|------------------------------------------------------------------|
| **UI**         | Gradio (Blocks API)                          | Web interface for chat, upload, convert, and translate            |
| **Backend**    | Python 3.10+                                 | Core application logic                                            |
| **Vector DB**  | Azure Cosmos DB (NoSQL + vector, posgresDB)  | Stores document chunks + embeddings, enables similarity search    |
| **Embeddings** | Azure OpenAI (text-embedding models)         | Converts text into numeric vectors                                |
| **LLM**        | Azure OpenAI (chat completions)              | Generates answers and translations                                |
| **PDF parsing**| pdfminer.six, (foundry, contentUnderstand)   | Extracts text from PDF documents                                  |
| **Config**     | python-dotenv (`dotenv_values`)              | Reads secrets and settings from `.env`                            |
| **Retry**      | tenacity                                     | Resilient retries for embedding/LLM calls                         |

---

## Component Breakdown

### 1. Configuration & Secrets (`config = dotenv_values()`)
Reads the `.env` file for all Azure credentials and settings:
- `cosmos_uri`, `cosmos_key`, `cosmos_database_name`, `cosmos_container_name`
- `cosmos_vector_property_name`, `cosmos_cache_database_name`, `cosmos_cache_container_name`
- `openai_endpoint`, `openai_key`, `openai_api_version`
- `openai_embeddings_deployment`, `openai_embeddings_dimensions`
- `openai_completions_deployment`
- `SOURCE_DIRECTORY_SOURCE` (default `source_documents`)

### Blob storage ingestion 
When a new/updated blob arrives (e.g., a normalized Markdown chunk or PDF), create/update a metadata record in the Cosmos container. Use Blob trigger decorator (Python v2 template). Event types like Microsoft.Storage.BlobCreated come with a payload that includes the blob URL. 


### 2. Azure Cosmos DB Setup
- Two containers are created (if they don't exist):
  - **CFTC container** — stores the actually ingested document chunks with their vector embeddings.
  - **Cache container** — stores chat history and cached Q&A pairs (with vectors) for fast repeat answers.
- Both containers use a **vector index policy** (`quantizedFlat`) and a **vector embedding policy**
  (cosine distance, `float32`). The partition key is `/id`.

### 3. Embedding Generation (`generate_embeddings`)
- Calls Azure OpenAI embeddings API.
- Uses `@retry` from tenacity to handle transient failures (exponential backoff, up to 20 attempts).

### 4. Document Ingestion (`ingest_documents`)
- Scans `source_documents/` recursively for `.pdf` and `.txt` files.
- For each file: extracts text (pdfminer for PDF, plain read for TXT).
- Splits text into **overlapping chunks** (`CHUNK_SIZE=1500`, `CHUNK_OVERLAP=200`).
- For each chunk: generates an embedding and **upserts** it into the CFTC container.
- Upsert-by-id makes the process **idempotent** — re-running doesn't duplicate documents.
- Runs automatically on startup.

### 5. Vector Search (`vector_search`)
- Queries Cosmos DB using `VectorDistance(c.vector, @embedding)` to find the most similar
  document chunks to the user's question. Returns the top matches with similarity scores.

### 6. Chat History (`get_chat_history`)
- Pulls the most recent completions from the cache container to provide conversational context.

### 7. Completion Generation (`generate_completion`)
- Builds a system prompt telling the model to answer **only** from the provided document
  context and to provide a list of at least 3 document answers where applicable.
- Feeds the user question, chat history, and retrieved document chunks to Azure OpenAI chat
  completions.

### 8. Caching (`cache_response` / `get_cache`)
- Stores the generated answer **and** its embeddings in the cache container.
- On a new question, first checks the cache with a very high similarity threshold (`0.99`).
  If a near-identical question was asked before, the cached answer is returned instantly.

### 9. Gradio UI
- **Chatbox** + **Enter button** + **Clear button**.
- **Upload & Ingest** section (file upload + button + status).
- **Document Converter** section (file upload + format dropdown + convert button + download).
- **Translator** section (input, source/target language dropdowns, translate button, output).

### 10. Document Converter (`convert_document`, `_text_to_format`)
- Converts text between formats using Python stdlib + installed libs (`yaml`, `csv`, `xml`,
  `html`, `email`). Produces a downloadable file.

### 11. Translator (`translate_text`)
- Uses the Azure OpenAI chat completions model with a translation system prompt.
- Supports auto-detection of source language.

### 12. Compatibility Patches
- Two patches are applied at import time to fix known incompatibilities between
  `gradio 4.44.1`, `gradio_client 1.3.0`, and `starlette 1.3.1`:
  1. `_json_schema_to_python_type` handles boolean schemas (avoids a 500 on `/config`).
  2. `Jinja2Templates.TemplateResponse` handles the old 2-arg signature (avoids a TypeError).

---

## Data Flow

```
 PDF/TXT files in source_documents/
        |
        |  ingested on startup (and on demand via Upload & Ingest)
        v
[extract_text] -> [split_text] -> [generate_embeddings] -> [upsert to Cosmos DB (CFTC container)]
        |
        v
User asks a question in the UI
        |
        v
[generate_embeddings(user question)]
        |
        +--> [get_cache: is it in cache?] --> YES --> return cached answer (fast)
        |
        | NO
        v
[vector_search: find similar chunks in CFTC container]
        |
        v
[get_chat_history: recent context]
        |
        v
[generate_completion: LLM answers from retrieved chunks]
        |
        v
[cache_response: store answer + embeddings]
        |
        v
Return answer to UI
```

---

## Sequence Diagram

```
User        Gradio UI        rag-chatbot.py          Azure OpenAI        Cosmos DB
 |              |                  |                       |                |
 |--question--->|                  |                       |                |
 |              |--enter/submit--> |                       |                |
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

---

## Architecture Design

- **Single-process app**: `rag-chatbot.py` runs the Gradio UI and all backend logic in one
  Python process.
- **Cloud services**: Azure OpenAI (embeddings + chat) and Azure Cosmos DB (vector store +
  cache) are external cloud services accessed via APIs.
- **Idempotent ingestion**: Documents are upserted by generated UUID, so re-running the app
  or re-ingesting is safe.
- **Two container separation**: The CFTC container holds the knowledge base; the cache
  container holds conversation history and cached answers — keeping concerns separate and
  improving cache hit latency.
- **Resilience**: tenacity retries smooth over transient network failures on embeddings and LLM calls.

---

## System Requirements

- **Python 3.10+** (3.10 or later recommended).
- **Azure subscription** with:
  - An Azure OpenAI resource (embeddings + chat completions deployments).
  - An Azure Cosmos DB account with **vector search** enabled.
- **Internet access** (cloud services are used).

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

# Optional
SOURCE_DIRECTORY_SOURCE=source_documents
```

---

## How To Run

```shell
python app.py
```

This will:
1. Load `.env` (Azure RAG credentials + Groq/Tavily keys).
2. Connect to Cosmos DB and create containers if needed.
3. Ingest all documents from `source_documents/` (if any).
4. Launch the unified Gradio UI at **http://127.0.0.1:7861** with two tabs:
   - **CFTC Document Q&A** (RAG chatbot)
   - **AI Agent (Web Search)** (Groq + Tavily)

> **Note:** The Groq/Tavily web-search agent requires `GROQ_API_KEY` and `TAVILY_API_KEY` to
> be set in `.env`. Without them, the app still launches and the RAG tab works, but the Agent
> tab will report a missing-key error.

You can also start both the FastAPI backend (`backend.py`) and the unified app via the
provided launcher:

```shell
python start_server.py   # launches app.py in the background (PID in server.pid)
```

---

## Usage Guide

### Asking Questions
1. Type a question in the **"Ask me about legal document content in the Database!"** box.
2. Press **Enter** or click the **Enter** button.
3. The assistant answers using the most relevant document chunks. Repeat questions are
   answered instantly from cache.

### Uploading & Ingesting Documents
1. In the **Upload & Ingest Documents** section, click **Choose documents** to select one or
   more PDF/TXT files.
2. Click **Upload & Ingest Documents**.
3. The files are saved to `source_documents/` and automatically ingested (chunked, embedded,
   and pushed to Cosmos DB). A status message shows how many items were ingested.

### Converting Documents
1. In the **Document Converter** section, upload a document.
2. Choose the target format from the **Convert to** dropdown.
3. Click **Convert Document**.
4. A downloadable converted file appears below.

### Translating Text
1. In the **Translator** section, enter the text to translate.
2. Choose the **source language** (or `auto`) and the **target language**.
3. Click **Translate**.
4. The translation appears in the output box.

---

## Smoke Test

To verify the app end-to-end non-interactively (server starts, root page loads, config API
works, and the chatbot answers both fresh and cached questions):

```shell
python rag-chatbot.py --smoke-test
```

This launches the app, runs checks, prints `PASS` or `FAIL`, and exits.

---

## Project Structure

```
cftc-gpt/
├── app.py                 # Unified Gradio application (2 tabs: RAG Q&A + AI Agent)
├── rag_core.py            # Shared RAG backend logic (Cosmos, embeddings, ingest, cache, convert, translate)
├── ai_agent.py            # Langgraph agent (Groq LLM + Tavily web search)
├── backend.py             # FastAPI backend exposing /chat, /health, /models
├── frontend.py            # Legacy Streamlit frontend for the agent (optional)
├── test_app.py            # pytest smoke tests
├── rag-chatbot.py         # Legacy standalone RAG Gradio app (kept for reference)
├── cftc.py                # Legacy Streamlit prototype (not used by the current app)
├── ingest.py              # Legacy local Chroma ingestion (not used by the current app)
├── constants.py           # Legacy Chroma settings
├── requirements.txt       # Python dependencies (RAG)
├── requirements.langgraph.txt  # Python dependencies (agent)
├── .env                   # Azure RAG secrets & configuration (not committed)
├── .env.langgraph         # model/model API key placeholders (not committed)
├── source_documents/      # Documents to ingest (PDF/TXT)
├── db/                    # Local Chroma data (legacy)
├── scripts/               # Dependency download helpers
├── README.md              # This documentation
└── TODO.md                # Task tracking
```



-------------------------------
Extract & chunk document (title, page numbers, original path).
Generate embeddings using Azure OpenAI (text-embedding-ada-002 → 1536 dimensions).
Create a Cosmos DB item containing:

id
title
page_number
source_path
chunk_text
vectors (float array)
timestamp

--
Upload original or normalized document to Blob Storage using the same source_path convention.
Store Blob SAS URL (or blob name) in Cosmos metadata for retrieval.

--PDF → JSON Extraction Options (concise)

Method 1 (Python / pdfplumber): Extract text/tables; assemble dict → json.dumps().
Method 2 (Node.js / pdf-parse): Buffer PDF → pdf() → parse lines → JSON.stringify().
Method 3 (PDF Vector Ask API): SDK client → client.ask() with mode:"json" + schema → structured JSON output.



5) Enhanced Citation (storage-backed)
To persist citations and show direct references:

Store chunk metadata (document title, page numbers, source path) alongside embeddings in Cosmos.
Save original documents (or normalized copies) in Blob Storage.
In responses, include: document name, page range, and Blob URL (SAS or via a private link gateway), so users can click through to the exact source.

Cosmos query: Vector search returns relevant chunks in <1s (cache path).
UI: / and /config return 200; /run/predict returns grounded answer.
Policy: Non-compliant count reduced after removing site extensions.




--------------------------------Agentic automation---

This project is a web-based AI chatbot application that leverages the Langgraph framework and the Groq LLM (Large Language Model) to provide intelligent responses to user queries. The chatbot is enhanced with a web search tool, Tavily, which allows it to fetch and present relevant information from the web.

## Key Features

- **AI Chatbot**: Utilizes the Groq LLM to generate responses to user queries based on a defined system prompt.
- **Web Search Integration**: Incorporates the Tavily search tool to fetch relevant web search results and present them in a user-friendly format.
- **Customizable System Prompt**: Allows users to define the behavior and personality of the AI agent through a customizable system prompt.
- **User-Friendly Interface**: Built with Streamlit, providing an intuitive and interactive user interface for querying the AI agent.
- **Formatted Responses**: Formats web search results to include titles, snippets, and links, making the information easy to read and navigate.

## Components

1. **Frontend (`frontend.py`)**:
   - Built with Streamlit to provide a user-friendly interface.
   - Allows users to input queries, select models, and define system prompts.
   - Displays the AI agent's responses, including formatted web search results.

2. **Backend (`backend.py`)**:
   - Built with FastAPI to handle API requests from the frontend.
   - Processes user queries and interacts with the AI agent.
   - Returns formatted responses to the frontend.

3. **AI Agent (`ai_agent.py`)**:
   - Utilizes the Groq LLM to generate responses.
   - Integrates the Tavily search tool to fetch web search results.
   - Formats the search results to include relevant information (titles, snippets, and links).

## How It Works

1. **User Interaction**: Users interact with the chatbot through the Streamlit interface, inputting queries and defining system prompts.
2. **API Request**: The frontend sends the user query and other parameters to the backend via an API request.
3. **AI Processing**: The backend processes the request using the Groq LLM and, if enabled, the Tavily search tool.
4. **Response Formatting**: The AI agent formats the web search results and generates a response.


---


== Production Tips (future TODOs_)===
create full Python pipeline script (ingest → chunk → OCR → embed → Cosmos → Blob).
Use Azure Blob Storage versioning + lifecycle policies.
Chunk text aggressively (300–500 tokens) to reduce embedding cost.
Store OCR + original document; often you need both later.
Use SAS tokens or Entra‑based access for secure doc access.
Wrap Cosmos calls with retry policies (especially on burst loads).
Cosmos → Cosmos sync: Use Azure Functions Cosmos DB trigger (internally backed by the change feed) to catch inserts/updates and upsert them into a second container—reliable, parallelized, and checkpointed via a lease container. 
Blob events → kick off ingestion: Use the event‑based Blob trigger (Event Grid) for near real‑time events on BlobCreated / BlobUpdated; this avoids polling and is required on Flex Consumption plans.
- reduce Azure Cosmos DB costs by ~95% through automatic archiving of infrequently accessed data to Blob Storage Cool Tier. Optimise cosmosDB by moving all archive, older retention data to Cold Blob Storage - https:///pt-vamshi/azure-task
- use SimpleChat or OpenWebUI for the frontend. 
- enable semantic re-ranker. Refine query and search results of any kind — full-text search, feed, vector search, or standard queries — using a semantic model to rerank items by semantic similarity to the query text. Applicable to any result set small enough to rerank. currently Inactive — reranking is not applied.Usage-based charges apply: $1 per 1,000 reranker calls. Regional prices may vary.


## Disclaimer

This is a test project to validate the feasibility of a RAG chatbot over CFTC legal documents.
It is **not production-ready**. Answers are generated by an AI model and should be reviewed by
a qualified professional before any legal reliance.
# Here is the detailed azure admin config guide for each component and infrastructure - https://cftcusgov-my.sharepoint.com/:w:/r/personal/nfletcher_cftc_gov/Documents/csl-usecase14-configuration.docx?d=wf39817a37eb94502894bbaffbb31b25a&csf=1&web=1&e=KCPAF9 
#https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/rag-chatbot