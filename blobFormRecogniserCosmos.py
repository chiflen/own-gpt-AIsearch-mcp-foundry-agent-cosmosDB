# -*- coding: utf-8 -*-
"""
blobFormRecogniserCosmos.py
===========================
Norman Fletcher
Azure Functions (Python v2) Blob-triggered pipeline that:

  1. Reacts to newly created / updated blobs in the Azure Blob Storage
     container ``csl14/csl-source`` (Blob trigger decorator, Python v2 template).
  2. Reads the blob (document) content.
  3. Calls Azure Document Intelligence (Form Recognizer) to extract text/content.
  4. Upserts a metadata record into a Cosmos DB container, including:
       - ``id``            unique document id
       - ``title``         document file name
       - ``source_path``   blob path (container/blob) for tracking
       - ``blob_url``      full blob URL so the RAG prompt response can link
                           back to the exact source document
       - ``container``     blob container name
       - ``blob_name``     blob name
       - ``chunk_text``    extracted text content
       - ``timestamp``     ISO timestamp
       - ``page_count``    number of pages detected by Form Recognizer
       - ``blob_size``     blob size in bytes

It also provides a ``sync_all_blobs`` (queue-triggered) entry point that scans
the whole source container and backfills/processes every existing blob so the
Cosmos container is kept in sync even for documents that were uploaded before
the trigger was enabled.

Event types like ``Microsoft.Storage.BlobCreated`` come with a payload that
includes the blob URL; the Blob trigger provides the blob name/path directly.

Requires the following packages:
    pip install azure-functions azure-ai-formrecognizer azure-storage-blob azure-cosmos python-dotenv
"""

import io
import os
import hashlib
import logging
import uuid
from datetime import datetime, timezone

# Reduce noisy Azure SDK HTTP logging (only show warnings+).
logging.getLogger("azure").setLevel(logging.WARNING)

import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# ---------------------------------------------------------------------
# Load configuration from .env (shared with the rest of the project)
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv, dotenv_values
    load_dotenv()
    _cfg = dotenv_values()
except Exception:  # pragma: no cover - dotenv may be unavailable
    _cfg = {}

app = func.FunctionApp()


def _get_env(key: str, default: str = "") -> str:
    """Read an env var from the process environment first, then .env values."""
    return os.getenv(key) or _cfg.get(key) or default


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BLOB_CONNECTION_STRING = _get_env(
    "blob_storage_connection_string",
)
BLOB_ACCOUNT_URL = _get_env(
    "blob_storage_url", "https://stgaicsldev.blob.core.windows.net"
)
BLOB_CONTAINER = _get_env("blob_storage_container_name", "csl14")
BLOB_SOURCE_PREFIX = _get_env("blob_folder_path", f"{BLOB_ACCOUNT_URL}/{BLOB_CONTAINER}/csl-source/")

FR_ENDPOINT = _get_env(
    "form_recognizer_endpoint", "https://docintel-csl14.cognitiveservices.azure.com/"
)
FR_KEY = _get_env("form_recognizer_key", "")
FR_MODEL_ID = _get_env("form_recognizer_model_id", "prebuilt-document")
FR_LANGUAGE = _get_env("form_recognizer_language", "en")
FR_TIMEOUT = int(_get_env("form_recognizer_timeout", "30") or 30)
FR_POLLING_INTERVAL = int(_get_env("form_recognizer_polling_interval", "5") or 5)

COSMOS_URI = _get_env("cosmos_uri", "")
COSMOS_KEY = _get_env("cosmos_key", "")
COSMOS_DATABASE = _get_env("cosmos_database_name", "SampleDB")
COSMOS_CONTAINER = _get_env("cosmos_container_name", "Cosmic-CFTC-Container")

PARTITION_KEY = "/source_path"

# ---------------------------------------------------------------------
# Clients (lazily initialised)
# ---------------------------------------------------------------------
_blob_service_client = None
_fr_client = None
_cosmos_client = None
_cosmos_database = None
_cosmos_container_client = None


def get_blob_service_client() -> BlobServiceClient:
    global _blob_service_client
    if _blob_service_client is None:
        if BLOB_CONNECTION_STRING:
            _blob_service_client = BlobServiceClient.from_connection_string(
                BLOB_CONNECTION_STRING
            )
        elif BLOB_ACCOUNT_URL:
            from azure.identity import DefaultAzureCredential
            _blob_service_client = BlobServiceClient(
                account_url=BLOB_ACCOUNT_URL, credential=DefaultAzureCredential()
            )
        else:
            raise ValueError("No blob storage connection string or account URL configured.")
    return _blob_service_client


def get_form_recognizer_client() -> DocumentAnalysisClient:
    global _fr_client
    if _fr_client is None:
        if not FR_ENDPOINT or not FR_KEY:
            raise ValueError("Form Recognizer endpoint/key not configured.")
        _fr_client = DocumentAnalysisClient(
            endpoint=FR_ENDPOINT,
            credential=AzureKeyCredential(FR_KEY),
        )
    return _fr_client


def get_cosmos_container():
    """Return the Cosmos container client, creating DB/container if needed."""
    global _cosmos_client, _cosmos_database, _cosmos_container_client
    if _cosmos_container_client is None:
        if not COSMOS_URI or not COSMOS_KEY:
            raise ValueError("Cosmos URI/key not configured.")
        _cosmos_client = CosmosClient(url=COSMOS_URI, credential=COSMOS_KEY)
        _cosmos_database = _cosmos_client.create_database_if_not_exists(COSMOS_DATABASE)
        try:
            _cosmos_container_client = _cosmos_database.get_container_client(
                COSMOS_CONTAINER
            )
            _cosmos_container_client.read()
        except exceptions.CosmosResourceNotFoundError:
            _cosmos_container_client = _cosmos_database.create_container_if_not_exists(
                id=COSMOS_CONTAINER,
                partition_key=PartitionKey(path=PARTITION_KEY),
                offer_throughput=400,
            )
    return _cosmos_container_client


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_blob_path(blob_path: str):
    """Split a blob path (container/name or URL) into (container, name, url)."""
    blob_path = blob_path.strip()

    # If it's a full URL, parse it.
    if blob_path.lower().startswith("http"):
        from urllib.parse import urlparse, quote
        parsed = urlparse(blob_path)
        parts = [p for p in parsed.path.split("/") if p]
        container = parts[0] if parts else BLOB_CONTAINER
        blob_name = "/".join(parts[1:])
        blob_url = blob_path
        return container, blob_name, blob_url

    # Otherwise treat first segment as container, rest as blob name.
    parts = blob_path.split("/", 1)
    container = parts[0] if parts else BLOB_CONTAINER
    blob_name = parts[1] if len(parts) > 1 else parts[0]
    from urllib.parse import quote
    blob_url = f"{BLOB_ACCOUNT_URL}/{container}/{quote(blob_name)}"
    return container, blob_name, blob_url


def make_item_id(container: str, blob_name: str) -> str:
    """Return a Cosmos-safe unique id for a blob.

    Cosmos DB ids cannot contain ``/``, ``\\``, ``?``, ``#``, or some other
    characters. We base the id on a stable hash of the source_path so it is both
    unique and idempotent (same blob => same id => upsert overwrites cleanly).
    """
    source_path = f"{container}/{blob_name}"
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:32]
    return f"blob-{digest}"


def process_blob(blob_name: str, container: str = BLOB_CONTAINER) -> dict:
    """Process a single blob: read -> OCR/Form Recognize -> upsert to Cosmos.

    Returns the Cosmos metadata item that was upserted.
    """
    logging.info(f"Processing blob: {container}/{blob_name}")

    if not blob_name:
        raise ValueError("Empty blob name provided.")

    source_path = f"{container}/{blob_name}"
    item_id = make_item_id(container, blob_name)

    # Only process documents we care about.
    ext = os.path.splitext(blob_name)[1].lower()
    if ext not in (".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        logging.info(f"Skipping non-document blob: {blob_name}")
        item = {
            "id": item_id,
            "title": os.path.basename(blob_name),
            "source_path": source_path,
            "blob_url": f"{BLOB_ACCOUNT_URL}/{container}/{blob_name}",
            "container": container,
            "blob_name": blob_name,
            "chunk_text": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blob_size": 0,
            "page_count": 0,
            "status": "skipped",
        }
        return item

    blob_service = get_blob_service_client()
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    props = blob_client.get_blob_properties()
    size = props.size
    blob_url = f"{BLOB_ACCOUNT_URL}/{container}/{blob_name}"

    # Download the blob content into memory.
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    stream.seek(0)
    data = stream.read()

    # Extract text via Azure Document Intelligence (Form Recognizer).
    content = ""
    page_count = 0
    try:
        fr_client = get_form_recognizer_client()
        poller = fr_client.begin_analyze_document(
            model_id=FR_MODEL_ID,
            document=data,
            polling_interval=FR_POLLING_INTERVAL,
        )
        result = poller.result()
        content = result.content or ""
        page_count = len(result.pages) if result.pages else 0
        logging.info(
            f"Form Recognizer extracted {len(content)} chars, "
            f"{page_count} pages for {blob_name}"
        )
    except Exception as e:
        logging.error(f"Form Recognizer failed for {blob_name}: {e}", exc_info=True)
        # Fall back to raw text attempt for text-like files.
        if ext in (".txt", ".csv"):
            try:
                content = data.decode("utf-8", errors="ignore")
            except Exception:
                content = ""

    # Build the Cosmos metadata item.
    item = {
        "id": item_id,
        "title": os.path.basename(blob_name),
        "source_path": source_path,
        "blob_url": blob_url,
        "container": container,
        "blob_name": blob_name,
        "chunk_text": content,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blob_size": size,
"page_count": page_count,
        "status": "processed",
    }

    container_client = get_cosmos_container()
    # The partition key path is "/source_path"; the value is read from the
    # body automatically, so no explicit partition_key argument is needed.
    container_client.upsert_item(body=item)
    logging.info(f"Upserted metadata for {container}/{blob_name} into Cosmos DB")
    return item


# ---------------------------------------------------------------------
# Blob trigger (Python v2 template) — fires on create/update
# ---------------------------------------------------------------------
@app.function_name(name="blob_trigger_ingest")
@app.blob_trigger(
    arg_name="blob",
    path="csl14/csl-source/{name}",
    connection="BlobStorageConnectionString",
)
def blob_trigger_ingest(blob: func.InputStream):
    """React to new/updated blobs in csl14/csl-source and sync to Cosmos."""
    logging.info(
        f"Python blob trigger function processed blob: {blob.name} "
        f"({blob.length} bytes)"
    )
    try:
        name = blob.name or ""
        # blob.name is like "csl14/csl-source/<file>" (container/blob)
        # Normalise to a blob path we can parse.
        if "/" in name and not name.lower().startswith("http"):
            # If it includes the container as first segment, parse it.
            container, blob_name, _ = parse_blob_path(name)
        else:
            container, blob_name = BLOB_CONTAINER, name
        process_blob(blob_name, container)
    except Exception as e:
        logging.error(f"Failed to process blob {blob.name}: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------
# Queue trigger — backfill / sync all existing blobs
# ---------------------------------------------------------------------
@app.function_name(name="sync_all_blobs")
@app.queue_trigger(
    arg_name="msg",
    queue_name="csl-source-sync",
    connection="AzureWebJobsStorage",
)
def sync_all_blobs(msg: func.QueueMessage):
    """Backfill/scan all blobs in the source container and process each.

    Trigger manually by sending any message to the ``csl-source-sync`` queue.
    """
    logging.info(f"sync_all_blobs triggered by message: {msg.get_body().decode('utf-8', 'ignore')}")
    blob_service = get_blob_service_client()
    container_client = blob_service.get_container_client(BLOB_CONTAINER)

    count = 0
    for blob in container_client.list_blobs(name_starts_with="csl-source/"):
        blob_name = blob.name
        try:
            process_blob(blob_name, BLOB_CONTAINER)
            count += 1
        except Exception as e:
            logging.error(f"Failed to process {blob_name}: {e}", exc_info=True)

    logging.info(f"sync_all_blobs complete. Processed {count} blobs.")


# ---------------------------------------------------------------------
# Manual / CLI entry point — run the backfill directly for testing
# ---------------------------------------------------------------------
def main():
    """Run the full sync of the source container from the command line."""
    logging.basicConfig(level=logging.INFO)
    blob_service = get_blob_service_client()
    container_client = blob_service.get_container_client(BLOB_CONTAINER)

    found = []
    for blob in container_client.list_blobs(name_starts_with="csl-source/"):
        found.append(blob.name)

    logging.info(f"Found {len(found)} blobs in {BLOB_CONTAINER}/csl-source/")
    count = 0
    for blob_name in found:
        try:
            item = process_blob(blob_name, BLOB_CONTAINER)
            count += 1
            logging.info(f"  OK -> {item['source_path']} ({item['blob_size']} bytes)")
        except Exception as e:
            logging.error(f"  FAIL -> {blob_name}: {e}", exc_info=True)
    logging.info(f"Backfill complete. Processed {count}/{len(found)} blobs.")


if __name__ == "__main__":
    main()
