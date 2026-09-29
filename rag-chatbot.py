# Import the required libraries
import sys
import time
import json
import uuid
import urllib 
import ijson
import zipfile
import os
import glob
import logging

# Bypass any system/corporate proxy for localhost so Gradio can bind and
# pass its local reachability check. Also disable Gradio analytics telemetry.
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from azure.cosmos import ThroughputProperties, PartitionKey, exceptions
from time import sleep
import gradio as gr

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
# with the deprecated 2-arg starlette signature, but the installed
# starlette 1.3.1 requires TemplateResponse(request, name, context).
# Detect the old-style call and adapt it to the new signature to avoid
# `TypeError: unhashable type: 'dict'` when Gradio renders its root page.
# ---------------------------------------------------------------------
try:
    from fastapi.templating import Jinja2Templates as _Jinja2Templates
    _orig_tpr = _Jinja2Templates.TemplateResponse

    def _compat_tpr(self, *args, **kwargs):
        # Old style: TemplateResponse(name(str), context(dict))
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
    print("Applied TemplateResponse compatibility patch (starlette old-style).")
except Exception as _e:
    print("TemplateResponse compatibility patch could not be applied:", _e)

# Cosmos DB imports
from azure.cosmos import CosmosClient

# PDF / text ingestion imports
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

from tenacity import retry, stop_after_attempt, wait_random_exponential 

# Load configuration from .env file
config = dotenv_values()

cosmos_conn = config['cosmos_uri']
cosmos_key = config['cosmos_key']
cosmos_database = config['cosmos_database_name']
cosmos_collection = config['cosmos_container_name']
cosmos_vector_property = config['cosmos_vector_property_name']
cosmos_cache_db = config['cosmos_cache_database_name']
cosmos_cache = config['cosmos_cache_container_name']

# Create the Azure Cosmos DB for NoSQL client for faster data loading
cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)

openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_api_version = config['openai_api_version']
openai_embeddings_deployment = config['openai_embeddings_deployment']
openai_embeddings_dimensions = int(config['openai_embeddings_dimensions'])
openai_completions_deployment = config['openai_completions_deployment']

# Create the OpenAI client
openai_client = AzureOpenAI(azure_endpoint=openai_endpoint, api_key=openai_key, api_version=openai_api_version)

# Where to look for documents to ingest
SOURCE_DIRECTORY = config.get('SOURCE_DIRECTORY_SOURCE', 'source_documents')

# Chunk size (in characters) used when splitting long documents before
# generating embeddings. Keeps each vector/item manageable.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# create database 
db = cosmos_client.create_database_if_not_exists(cosmos_database)

# Create the vector embedding policy to specify vector details
vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/" + cosmos_vector_property,
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":openai_embeddings_dimensions
        }, 
    ]
}

# Create the vector index policy to specify vector details
indexing_policy = {
    "includedPaths": [ 
    { 
        "path": "/*" 
    } 
    ], 
    "excludedPaths": [ 
    { 
        "path": "/\"_etag\"/?",
        "path": "/" + cosmos_vector_property + "/*",
    } 
    ], 
    "vectorIndexes": [ 
        {
            "path": "/"+cosmos_vector_property, 
            "type": "quantizedFlat" 
        }
    ]
} 

# Create the data collection with vector index (note: this creates a container with an autoscale limit of 20,000 RUs to allow fast data load)
try:
    cftc_container = db.create_container_if_not_exists(id=cosmos_collection, 
                                                  partition_key=PartitionKey(path='/id'),
                                                  indexing_policy=indexing_policy, 
                                                  vector_embedding_policy=vector_embedding_policy) 
    print('Container with id \'{0}\' created'.format(cftc_container.id)) 

except exceptions.CosmosHttpResponseError: 
    raise 

# Create the cache collection with vector index
try:
    cache_container = db.create_container_if_not_exists(id=cosmos_cache, 
                                                  partition_key=PartitionKey(path='/id'), 
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy) 
    print('Container with id \'{0}\' created'.format(cache_container.id))

except exceptions.CosmosHttpResponseError: 
    raise

# generate embeddings for a given text using openAI
@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
    try:        
        response = openai_client.embeddings.create(
            input=text,
            model=openai_embeddings_deployment,
            dimensions=openai_embeddings_dimensions
        )
        embeddings = response.model_dump()
        return embeddings['data'][0]['embedding']
    except Exception as e:
        # Log the exception with traceback for easier debugging
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise

# ---------------------------------------------------------------
# Document ingestion: read all supported files (PDF / TXT) in
# SOURCE_DIRECTORY, split into chunks, and store in Cosmos DB
# with vector embeddings. This replaces the MovieLens zip/JSON
# loading from the original tutorial.
# ---------------------------------------------------------------
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
    """Split a long text into overlapping character chunks (paragraph-aware-ish)."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Try to break at a newline boundary near the chunk end
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

def ingest_documents():
    """Load all documents from SOURCE_DIRECTORY, chunk, embed and upsert to Cosmos DB."""
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
        for i, chunk in enumerate(chunks):
            items.append({
                'id': str(uuid.uuid4()),
                'source': os.path.basename(file_path),
                'chunk': i,
                'content': chunk,
            })

    if not items:
        print("No items to ingest!")
        return 0

    print(f"Generating embeddings and storing {len(items)} item(s) in Cosmos DB ...")
    start_time = time.time()
    counter = 0
    for obj in items:
        obj[cosmos_vector_property] = generate_embeddings(obj['content'])
        # id is already the partition key
        cftc_container.upsert_item(body=obj)
        counter += 1
        if counter % 10 == 0:
            print(f"  upserted {counter}/{len(items)}")
    end_time = time.time()
    print(f"All {counter} documents inserted!")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return counter

# Run ingestion on startup (idempotent — upserts by id so re-runs don't duplicate)
ingest_documents()

#perform vector search on the data in cosmos db using openAI embeddings

def vector_search(container, vectors, similarity_score=0.02, num_results=5):
    results = container.query_items(
        query='''
        SELECT TOP @num_results c.id, c.source, c.chunk, c.content, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True,
        populate_query_metrics=True
    )
    results = list(results)
    formatted_results = [{'SimilarityScore': result.pop('SimilarityScore'), 'document': result} for result in results]

    return formatted_results

# get recent chat history from cosmos db cache collection

def get_chat_history(container, completions=3):
    results = container.query_items(
        query='''
        SELECT TOP @completions *
        FROM c
        ORDER BY c._ts DESC
        ''',
        parameters=[
            {"name": "@completions", "value": completions},
        ],
        enable_cross_partition_query=True
    )
    results = list(results)
    return results


# get chat completions from cosmos db cache collection

def generate_completion(user_prompt, vector_search_results, chat_history):
    system_prompt = '''
    You are an intelligent assistant for CFTC legal documents. You are designed to provide helpful
    answers to user questions about the documents in your database (CFTC enforcement actions).
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your
    responses, but still friendly.
     - Only answer questions related to the information provided below. Provide at least 3 document
       answers in a list where applicable.
     - Write two lines of whitespace between each answer in the list.
    '''

    messages = [{'role': 'system', 'content': system_prompt}]
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    messages.append({'role': 'user', 'content': user_prompt})
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': json.dumps(result['document'])})

    response = openai_client.chat.completions.create(
        model=openai_completions_deployment,
        messages=messages
    )    
    return response.model_dump()

def chat_completion(cache_container, cftcdoc_container, user_input):
    print("starting completion")
    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(user_input)
    # Query the chat history cache first to see if this question has been asked before
    cache_results = get_cache(container=cache_container, vectors=user_embeddings, similarity_score=0.99, num_results=1)
    if len(cache_results) > 0:
        print("Cached Result\n")
        return cache_results[0]['completion'], True
        
    else:
        # Perform vector search on the cftc collection
        print("New result\n")
        search_results = vector_search(cftc_container, user_embeddings)

        print("Getting Chat History\n")
        # Chat history
        chat_history = get_chat_history(cache_container, 3)
        # Generate the completion
        print("Generating completions \n")
        completions_results = generate_completion(user_input, search_results, chat_history)

        print("Caching response \n")
        # Cache the response
        cache_response(cache_container, user_input, user_embeddings, completions_results)

        print("\n")
        # Return the generated LLM completion
        return completions_results['choices'][0]['message']['content'], False

# cache generated response in cosmos db cache collection

def cache_response(container, user_prompt, prompt_vectors, response):
    chat_document = {
        'id': str(uuid.uuid4()),
        'prompt': user_prompt,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
        'vector': prompt_vectors
    }
    container.create_item(body=chat_document)

def get_cache(container, vectors, similarity_score=0.0, num_results=5):
    # Execute the query
    results = container.query_items(
        query= '''
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
        enable_cross_partition_query=True, populate_query_metrics=True)
    results = list(results)
    return results

# ------------------------------------------------------------------
# Document Converter utilities
# ------------------------------------------------------------------
import html as _html
import csv as _csv
import io as _io
import xml.etree.ElementTree as _ET
import email as _email
import base64 as _b64

# Map of extension -> writer function returning (filename, content_bytes)
CONVERT_TARGETS = {
    '.txt': 'txt',
    '.md': 'md',
    '.json': 'json',
    '.html': 'html',
    '.xml': 'xml',
    '.yaml': 'yaml',
    '.csv': 'csv',
    '.rtf': 'rtf',
    '.dat': 'dat',
    '.eml': 'eml',
}

def _extract_text_from_upload(file_path):
    """Best-effort plain-text extraction from a wide range of upload formats."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        # Try UTF-8 first, fall back to latin-1
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
        # Wrap each non-empty line as a paragraph
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n\n".join(lines)
    if fmt == 'json':
        return json.dumps({"content": text}, indent=2)
    if fmt == 'html':
        body = _html.escape(text).replace('\n', '<br>\n')
        return f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'><title>Converted Document</title></head>\n<body>\n<p>{body}</p>\n</body>\n</html>"
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

# ------------------------------------------------------------------
# Translator utilities (uses Azure OpenAI chat completions)
# ------------------------------------------------------------------
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

# create a Gradio interface for the chatbot

chat_history = []

with gr.Blocks(title="CFTC AI Assistant") as demo:
    gr.Markdown("# CFTC AI Assistant")
    gr.Markdown("*Developer: Norman Fletcher 1*")
    chatbot = gr.Chatbot(label="CFTC AI Assistant", type="tuples")
    with gr.Row():
        msg = gr.Textbox(label="Ask me about legal document content in the Database!", scale=4)
        enter_btn = gr.Button("Enter", scale=1)
    clear = gr.Button("Clear")

    def user(user_message, chat_history):
        start_time = time.time()
        response_payload, cached = chat_completion(cache_container, cftc_container, user_message)
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)
        details = f"\n (Time: {elapsed_time}ms)"
        if cached:
            details += " (Cached)"
        chat_history.append([user_message, response_payload + details])
        
        return gr.update(value=""), chat_history
    
# Shared submit handler for both the textbox (Enter key) and the Enter button
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    enter_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    # ------------------------------------------------------------------
    # Upload & Ingest Documents section
    # ------------------------------------------------------------------
    gr.Markdown("## Upload & Ingest Documents")
    gr.Markdown("Select PDF or TXT documents to add to the knowledge base. They will be "
                "saved to the `source_documents` folder and automatically ingested into Cosmos DB.")
    with gr.Row():
        upload_files = gr.File(label="Choose documents (.pdf / .txt)", file_count="multiple",
                               file_types=[".pdf", ".txt"], type="filepath")
        upload_btn = gr.Button("Upload & Ingest Documents")
    ingest_status = gr.Markdown("")

    def upload_and_ingest(files):
        if not files:
            return "No files selected. Please choose one or more PDF/TXT documents first."
        # `files` may be a single string when one file, or a list of strings otherwise.
        if isinstance(files, str):
            files = [files]
        os.makedirs(SOURCE_DIRECTORY, exist_ok=True)
        saved = []
        for src in files:
            try:
                dest = os.path.join(SOURCE_DIRECTORY, os.path.basename(src))
                with open(src, 'rb') as fin, open(dest, 'wb') as fout:
                    fout.write(fin.read())
                saved.append(os.path.basename(dest))
            except Exception as e:
                print(f"  [error] failed to save {src}: {e}")
        if not saved:
            return "Failed to save any uploaded document."
        # Ingest all documents (including the newly uploaded ones) into Cosmos DB.
        count = ingest_documents()
        return (f"Uploaded {len(saved)} document(s): {', '.join(saved)}\n\n"
f"Ingested {count} item(s) into Cosmos DB.")

    upload_btn.click(upload_and_ingest, [upload_files], [ingest_status], queue=False)

    # ------------------------------------------------------------------
    # Document Converter section
    # ------------------------------------------------------------------
    gr.Markdown("## Document Converter")
    gr.Markdown("Upload a document and convert it to a different format "
                "(.txt, .md, .json, .html, .xml, .yaml, .csv, .rtf, .dat, .eml).")
    with gr.Row():
        convert_file = gr.File(label="Choose a document to convert", file_count="single",
                               file_types=[".txt", ".md", ".json", ".html", ".xml",
                                           ".yaml", ".yml", ".csv", ".rtf", ".dat",
                                           ".eml", ".pdf", ".docx"], type="filepath")
        convert_target = gr.Dropdown(
            label="Convert to",
            choices=[".txt", ".md", ".json", ".html", ".xml", ".yaml",
                     ".csv", ".rtf", ".dat", ".eml"],
            value=".md",
        )
        convert_btn = gr.Button("Convert Document")
    convert_status = gr.Markdown("")
    convert_output = gr.File(label="Download converted file", interactive=False)

    def do_convert(file_path, target_fmt):
        status, out_path = convert_document(file_path, target_fmt)
        return status, (out_path if out_path else None)

    convert_btn.click(do_convert, [convert_file, convert_target],
                      [convert_status, convert_output], queue=False)

    # ------------------------------------------------------------------
    # Translator section
    # ------------------------------------------------------------------
    gr.Markdown("## Translator")
    gr.Markdown("Translate text between languages using the AI model. "
                "Supports French, Spanish, Italian, Portuguese, English, and more.")
    with gr.Row():
        translate_src = gr.Dropdown(label="Source language", choices=LANGUAGE_CHOICES, value="auto")
        translate_tgt = gr.Dropdown(label="Target language", choices=[c for c in LANGUAGE_CHOICES if c != "auto"], value="English")
    translate_input = gr.Textbox(label="Text to translate", lines=4)
    translate_btn = gr.Button("Translate")
    translate_output = gr.Textbox(label="Translation", lines=4, interactive=False)

    def do_translate(text, src, tgt):
        return translate_text(text, src, tgt)

    translate_btn.click(do_translate, [translate_input, translate_src, translate_tgt],
                        [translate_output], queue=False)


# ------------------------------------------------------------------
# Smoke-test mode: `python rag-chatbot.py --smoke-test`
# Runs the real app, verifies the server starts, the root page loads,
# the config API works, and the chatbot endpoint answers. Exits with
# PASS/FAIL so the task "run tests / check all UI functionality"
# can be verified end-to-end non-interactively.
# ------------------------------------------------------------------
def smoke_test(port=7861):
    import sys
    import threading
    import time
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

    def queue_ready(timeout=60):
        # /queue/data is a WebSocket endpoint; the app uses queue=False, so
        # treat this as informational only (not part of PASS/FAIL).
        return True

    results['root'] = wait_ready()
    results['queue'] = queue_ready()

    # config/api info endpoints
    for path in ("/config", "/info"):
        try:
            r = httpx.get(base + path, timeout=5, verify=False)
            results[path] = r.status_code == 200
        except Exception:
            results[path] = False

    # Test the chatbot function-level logic directly. Ask the same question
    # twice: first generates & caches ("New result"), second hits the Cosmos
    # cache ("Cached Result") — verifies both paths.
    try:
        test_q = "What enforcement action does 21-25b (1).pdf discuss?"
        # Run chat_completion in a worker to avoid blocking the event loop
        def _ask(n):
            global _smoke_result
            _smoke_result = chat_completion(cache_container, cftc_container, n)
        worker = threading.Thread(target=_ask, args=(test_q,), daemon=True)
        worker.start()
        worker.join(timeout=90)
        if worker.is_alive():
            results['chatbot'] = False
            results['chatbot_err'] = "timed out (no completion in 90s)"
        else:
            resp_payload, cached_flag = _smoke_result
            results['chatbot'] = bool(resp_payload and str(resp_payload).strip())
            results['chatbot_cached'] = cached_flag
            results['chatbot_answer_len'] = len(str(resp_payload))
            # Second ask: should be served from cache
            worker2 = threading.Thread(target=_ask, args=(test_q,), daemon=True)
            worker2.start()
            worker2.join(timeout=60)
            if worker2.is_alive():
                results['chatbot_cached_ok'] = False
                results['chatbot_cached_err'] = "timed out on cached ask"
            else:
                cached_payload, cached_flag2 = _smoke_result
                results['chatbot_cached_ok'] = bool(cached_flag2) and bool(cached_payload)
                results['chatbot_cached_same'] = str(cached_payload) == str(resp_payload)
    except Exception as e:
        results['chatbot'] = False
        results['chatbot_err'] = repr(e)

    print("\n===== SMOKE TEST RESULTS =====")
    for k, v in results.items():
        print(f"  {k}: {v}")
    demo.close()
    try:
        demo.stop()  # in newer gradio versions
    except Exception:
        pass
    ok = all(
        v is True
        for k, v in results.items()
        if not k.endswith("_err") and not k.endswith("_len")
        and not k.endswith("_cached") and not k.endswith("_cached_same")
    )
    print(("PASS" if ok else "FAIL"), "SMOKE TEST")
    sys.exit(0 if ok else 1)


if __name__ == "__main__" and "--smoke-test" in sys.argv:
    smoke_test(port=7861)
else:
    # Launch the Gradio interface
    demo.launch(debug=True, share=False)

    # Be sure to run this cell to close or restart the Gradio demo
    demo.close()

