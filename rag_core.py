# -*- coding: utf-8 -*-
"""
rag_core.py — shared backend logic for the CFTC RAG assistant.

Extracted from `rag-chatbot.py` so that both the standalone Gradio app and the
unified merged app (`app.py`) can reuse the same Cosmos DB / Azure OpenAI logic:
document ingestion, vector search, chat completion with caching, document
conversion and translation.
"""

import os
import sys
import time
import json
import uuid
import glob
import logging

# ---------------------------------------------------------------------
# Bypass any system/corporate proxy for localhost so Gradio can bind and
# pass its local reachability check. Also disable Gradio analytics telemetry.
# ---------------------------------------------------------------------
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# ---------------------------------------------------------------------
# Compatibility patch: gradio 4.44.1 is incompatible with gradio_client
# 1.3.0. When Gradio serializes its app config, gradio_client's
# _json_schema_to_python_type crashes on `additionalProperties: true`
# (a boolean schema), which makes the root endpoint return 500 and
# makes Gradio's `url_ok` localhost check fail. Patch it to handle
# non-dict schemas gracefully (proxy/network blocks a pip upgrade).
# ---------------------------------------------------------------------
import gradio_client.utils as _gcu
_orig_json_schema_to_python_type = _gcu._json_schema_to_python_type
def _safe_json_schema_to_python_type(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)
_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

# ---------------------------------------------------------------------
# Compatibility patch #2: gradio 4.44.1 calls TemplateResponse(name, context)
# with the deprecated 2-arg starlette signature, but newer starlette versions
# require TemplateResponse(request, name, context). Detect the old-style call
# and adapt it to the new signature to avoid `TypeError` when Gradio renders
# its root page.
# ---------------------------------------------------------------------
try:
    from fastapi.templating import Jinja2Templates as _Jinja2Templates
    _orig_tpr = _Jinja2Templates.TemplateResponse

    def _compat_tpr(self, *args, **kwargs):
        if (
            len(args) >= 2
            and isinstance(args[0], str)
            and isinstance(args[1], dict)
        ):
            name, context = args[0], args[1]
            request = context.get("request")
            if request is None:
                from starlette.requests import Request as _StarletteRequest

                request = _StarletteRequest(
                    {
                        "type": "http",
                        "method": "GET",
                        "path": "/",
                        "headers": [],
                        "query_string": b"",
                        "server": ("localhost", 80),
                        "client": ("127.0.0.1", 1234),
                        "scheme": "http",
                    }
                )
            return _orig_tpr(self, request, name, context, *args[2:], **kwargs)
        return _orig_tpr(self, *args, **kwargs)

    _Jinja2Templates.TemplateResponse = _compat_tpr
except Exception as _e:  # pragma: no cover
    print("TemplateResponse compatibility patch could not be applied:", _e)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ---------------------------------------------------------------------
# Configuration (.env)
# ---------------------------------------------------------------------
config = dotenv_values()

cosmos_conn = config['cosmos_uri']
cosmos_key = config['cosmos_key']
cosmos_database = config['cosmos_database_name']
cosmos_collection = config['cosmos_container_name']
cosmos_vector_property = config['cosmos_vector_property_name']
cosmos_cache_db = config['cosmos_cache_database_name']
cosmos_cache = config['cosmos_cache_container_name']

openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_api_version = config['openai_api_version']
openai_embeddings_deployment = config['openai_embeddings_deployment']
openai_embeddings_dimensions = int(config['openai_embeddings_dimensions'])
openai_completions_deployment = config['openai_completions_deployment']

openai_client = AzureOpenAI(
    azure_endpoint=openai_endpoint,
    api_key=openai_key,
    api_version=openai_api_version,
)

SOURCE_DIRECTORY = config.get('SOURCE_DIRECTORY_SOURCE', 'source_documents')
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Blob storage configuration (for source links in prompt responses)
BLOB_ACCOUNT_URL = config.get('blob_storage_url', 'https://stgaicsldev.blob.core.windows.net')
BLOB_CONTAINER = config.get('blob_storage_container_name', 'csl14')
BLOB_SOURCE_PREFIX = config.get('blob_folder_path', f'{BLOB_ACCOUNT_URL}/{BLOB_CONTAINER}/csl-source/')


def build_blob_url(source_filename):
    """Build the blob storage URL for a given source file name.

    The blob URL points to the source file in the blob container so the model
    can include a clickable link in its response.
    """
    if not source_filename:
        return ""
    # Ensure the prefix ends with a slash and the filename is URL-safe.
    prefix = BLOB_SOURCE_PREFIX.rstrip('/') + '/'
    from urllib.parse import quote
    safe_name = quote(source_filename)
    return f"{prefix}{safe_name}"

# ---------------------------------------------------------------------
# Cosmos DB containers (vector-enabled) — LAZY initialization.
#
# The Cosmos client & containers are created only on first use (via the
# `get_cosmos_container` / `get_cftc_container` / `get_cache_container`
# helpers below) rather than at import time. This makes the module (and
# therefore `app.py` / `ai_agent.py` / `backend.py` / the test suite) robust
# to transient network/DNS outages, because simply importing rag_core no
# longer performs any network I/O.
# ---------------------------------------------------------------------

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/" + cosmos_vector_property,
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": openai_embeddings_dimensions,
        },
    ]
}

indexing_policy = {
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [
        {"path": "/\"_etag\"/?"},
        {"path": "/" + cosmos_vector_property + "/*"},
    ],
    "vectorIndexes": [
        {"path": "/" + cosmos_vector_property, "type": "quantizedFlat"}
    ],
}


# Module-level singletons (populated lazily).
_cosmos_client = None
_db = None


def get_cosmos_client():
    """Return (and cache) the Cosmos DB client. Created on first use."""
    global _cosmos_client
    if _cosmos_client is None:
        _cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)
    return _cosmos_client


def get_database():
    """Return (and cache) the Cosmos database, creating it if needed."""
    global _db
    if _db is None:
        _db = get_cosmos_client().create_database_if_not_exists(cosmos_database)
    return _db


def _get_or_create_container(container_id):
    db = get_database()
    if container_id in [c['id'] for c in db.list_containers()]:
        container = db.get_container_client(container_id)
        # Ensure vector policy is present (required for VectorDistance queries)
        container.read()
        return container
    return db.create_container_if_not_exists(
        id=container_id,
        partition_key=PartitionKey(path='/id'),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )


def get_cftc_container():
    """Return (and cache) the CFTC document container, creating it if needed."""
    return _get_or_create_container(cosmos_collection)


def get_cache_container():
    """Return (and cache) the cache container, creating it if needed."""
    return _get_or_create_container(cosmos_cache)


class _LazyContainer:
    """A lazy proxy for a Cosmos container.

    Defers connection to Azure Cosmos DB until the first attribute access or
    method call, so that merely importing ``rag_core`` performs no network
    I/O. This keeps the app importable and startable even during transient
    network/DNS outages, and the real container is created lazily on first use.
    """

    def __init__(self, factory):
        self._factory = factory
        self._container = None

    def _resolve(self):
        if self._container is None:
            self._container = self._factory()
        return self._container

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __getitem__(self, item):
        return self._resolve()[item]

    def __iter__(self):
        return iter(self._resolve())

    @property
    def id(self):
        return self._resolve().id


# Public module-level aliases used by the rest of the codebase. These are lazy
# proxies, so importing ``rag_core`` (and thus ``app.py`` / ``ai_agent.py`` /
# ``backend.py``) performs no network I/O at import time.
cftc_container = _LazyContainer(get_cftc_container)
cache_container = _LazyContainer(get_cache_container)


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=openai_embeddings_deployment,
            dimensions=openai_embeddings_dimensions,
        )
        embeddings = response.model_dump()
        return embeddings['data'][0]['embedding']
    except Exception:
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise


# ---------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------
SUPPORTED_EXTENSIONS = ('.pdf', '.txt')


def extract_text_from_file(file_path):
    """Extract raw text from a PDF or TXT file."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return extract_text(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except PDFSyntaxError as e:
        print(f"  [skip] could not parse PDF {file_path}: {e}")
        return None
    except Exception as e:
        print(f"  [skip] could not read {file_path}: {e}")
        return None
    return None


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a long text into overlapping character chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            last_nl = text.rfind('\n', start, end)
            if last_nl > start + chunk_size // 2:
                end = last_nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_items():
    """Scan SOURCE_DIRECTORY and build (content) chunks to ingest."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(SOURCE_DIRECTORY, f"**/*{ext}"), recursive=True))

    print(f"Found {len(files)} document(s) in {SOURCE_DIRECTORY}")
    items = []
    for file_path in files:
        print(f"Reading {file_path} ...")
        text = extract_text_from_file(file_path)
        if not text or not text.strip():
            print(f"  [skip] no text extracted from {file_path}")
            continue
        chunks = split_text(text)
        print(f"  -> {len(chunks)} chunk(s)")
        filename = os.path.basename(file_path)
        blob_url = build_blob_url(filename)
        for i, chunk in enumerate(chunks):
            items.append({
                'id': str(uuid.uuid4()),
                'source': filename,
                'source_path': os.path.relpath(file_path, SOURCE_DIRECTORY),
                'blob_url': blob_url,
                'chunk': i,
                'content': chunk,
            })
    return items


def ingest_documents():
    """Load documents, chunk, embed, and upsert to Cosmos DB (idempotent)."""
    items = build_items()
    if not items:
        print("No items to ingest!")
        return 0

    print(f"Generating embeddings and storing {len(items)} item(s) in Cosmos DB ...")
    start_time = time.time()
    counter = 0
    for obj in items:
        obj[cosmos_vector_property] = generate_embeddings(obj['content'])
        cftc_container.upsert_item(body=obj)
        counter += 1
        if counter % 10 == 0:
            print(f"  upserted {counter}/{len(items)}")
    end_time = time.time()
    print(f"All {counter} documents inserted! Time taken: {end_time - start_time:.2f}s")
    return counter


# ---------------------------------------------------------------------
# Vector search + chat completion + caching
# ---------------------------------------------------------------------
def vector_search(container, vectors, similarity_score=0.02, num_results=5):
    results = container.query_items(
        query='''
        SELECT TOP @num_results c.id, c.source, c.source_path, c.blob_url,
               c.chunk, c.content,
               VectorDistance(c.vector, @embedding) as SimilarityScore
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score},
        ],
        enable_cross_partition_query=True,
        populate_query_metrics=True,
    )
    results = list(results)
    formatted_results = [
        {'SimilarityScore': result.pop('SimilarityScore'), 'document': result}
        for result in results
    ]
    return formatted_results


def get_chat_history(container, completions=3):
    results = container.query_items(
        query='''
        SELECT TOP @completions *
        FROM c
        ORDER BY c._ts DESC
        ''',
        parameters=[{"name": "@completions", "value": completions}],
        enable_cross_partition_query=True,
    )
    return list(results)


def generate_completion(user_prompt, vector_search_results, chat_history):
    system_prompt = '''
    You are an intelligent assistant for CFTC legal documents. You are designed to provide helpful
    answers to user questions about the documents in your database (CFTC enforcement actions).
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your
    responses, but still friendly.
     - Only answer questions related to the information provided below. Provide at least 3 document
       answers in a list where applicable.
     - Write two lines of whitespace between each answer in the list.
     - For each answer, include the source document citation with a clickable link using the
       `blob_url` field when present (e.g. `[document title](blob_url)`). If no blob_url is
       available, include the document title and source_path so the user can locate the source.
    '''

    messages = [{'role': 'system', 'content': system_prompt}]
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    messages.append({'role': 'user', 'content': user_prompt})
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': json.dumps(result['document'])})

    response = openai_client.chat.completions.create(
        model=openai_completions_deployment,
        messages=messages,
    )
    return response.model_dump()


def cache_response(container, user_prompt, prompt_vectors, response):
    chat_document = {
        'id': str(uuid.uuid4()),
        'prompt': user_prompt,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
        'vector': prompt_vectors,
    }
    container.create_item(body=chat_document)


def get_cache(container, vectors, similarity_score=0.0, num_results=5):
    results = container.query_items(
        query='''
        SELECT TOP @num_results *
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score},
        ],
        enable_cross_partition_query=True,
        populate_query_metrics=True,
    )
    return list(results)


def chat_completion(cache_container, cftcdoc_container, user_input):
    print("starting completion")
    user_embeddings = generate_embeddings(user_input)
    cache_results = get_cache(
        container=cache_container, vectors=user_embeddings,
        similarity_score=0.99, num_results=1,
    )
    if len(cache_results) > 0:
        print("Cached Result\n")
        return cache_results[0]['completion'], True

    print("New result\n")
    search_results = vector_search(cftcdoc_container, user_embeddings)

    print("Getting Chat History\n")
    chat_history = get_chat_history(cache_container, 3)

    print("Generating completions \n")
    completions_results = generate_completion(user_input, search_results, chat_history)

    print("Caching response \n")
    cache_response(cache_container, user_input, user_embeddings, completions_results)

    print("\n")
    return completions_results['choices'][0]['message']['content'], False


# ---------------------------------------------------------------------
# Document Converter utilities
# ---------------------------------------------------------------------
import html as _html
import csv as _csv
import io as _io
import xml.etree.ElementTree as _ET
import email as _email

CONVERT_TARGETS = {
    '.txt': 'txt', '.md': 'md', '.json': 'json', '.html': 'html',
    '.xml': 'xml', '.yaml': 'yaml', '.csv': 'csv', '.rtf': 'rtf',
    '.dat': 'dat', '.eml': 'eml',
}


def _extract_text_from_upload(file_path):
    """Best-effort plain-text extraction from a wide range of upload formats."""
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')
    except Exception as e:
        print(f"  [convert] failed to read {file_path}: {e}")
        return None


def _text_to_format(text, fmt):
    """Convert plain text into the requested target format."""
    if fmt == 'txt':
        return text
    if fmt == 'md':
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n\n".join(lines)
    if fmt == 'json':
        return json.dumps({"content": text}, indent=2)
    if fmt == 'html':
        body = _html.escape(text).replace('\n', '<br>\n')
        return (f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>"
                f"<title>Converted Document</title></head>\n<body>\n<p>{body}</p>\n"
                f"</body>\n</html>")
    if fmt == 'xml':
        root = _ET.Element('document')
        root.text = _html.escape(text)
        return _ET.tostring(root, encoding='unicode')
    if fmt == 'yaml':
        import yaml
        return yaml.safe_dump({"content": text}, allow_unicode=True, default_flow_style=False)
    if fmt == 'csv':
        buf = _io.StringIO()
        writer = _csv.writer(buf)
        for line in text.splitlines():
            writer.writerow([line])
        return buf.getvalue()
    if fmt == 'rtf':
        safe = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
        safe = safe.replace('\n', '\\par\n')
        return r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Courier New;}} \f0\fs24 " + safe + "\n}"
    if fmt == 'dat':
        return text
    if fmt == 'eml':
        msg = _email.message_from_string(
            "From: cftc-assistant@localhost\n"
            "To: user@localhost\n"
            "Subject: Converted Document\n"
            "MIME-Version: 1.0\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            + text
        )
        return msg.as_string()
    return text


def convert_document(file_path, target_fmt):
    """Convert an uploaded document to the requested format."""
    if not file_path:
        return "No file selected. Please upload a document first.", None
    text = _extract_text_from_upload(file_path)
    if not text:
        return "Could not read the uploaded file.", None
    base = os.path.basename(file_path)
    stem = os.path.splitext(base)[0]
    out_name = f"{stem}{target_fmt}"
    converted = _text_to_format(text, target_fmt.lstrip('.'))
    out_path = os.path.join(os.path.dirname(file_path) or '.', out_name)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(converted)
    except Exception as e:
        return f"Error writing converted file: {e}", None
    return (
        f"Converted **{base}** to **{out_name}** ({len(converted)} chars).",
        out_path,
    )


# ---------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------
LANGUAGE_CHOICES = [
    "auto", "English", "French", "Spanish", "Italian", "Portuguese",
    "German", "Dutch", "Chinese", "Japanese", "Arabic", "Russian", "Korean",
]


def translate_text(text, source_lang, target_lang):
    """Translate text using the configured Azure OpenAI completion model."""
    if not text or not text.strip():
        return "Please enter some text to translate."
    if target_lang == "auto":
        target_lang = "English"
    src = "auto-detected" if source_lang == "auto" else source_lang
    system_prompt = (
        "You are a professional translator. Translate the user's text accurately, "
        "preserving meaning, tone, and formatting. Output ONLY the translation, "
        "with no extra commentary."
    )
    user_prompt = (
        f"Detect the source language ({src}) and translate the text below into "
        f"{target_lang}. Return only the translated text.\n\n---\n{text}\n---"
    )
    try:
        response = openai_client.chat.completions.create(
            model=openai_completions_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.model_dump()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  [translate] error: {e}")
        return f"Translation failed: {e}"

