"""Verify the blob->Cosmos upsert worked by querying the container."""
import logging
logging.basicConfig(level=logging.INFO)
import blobFormRecogniserCosmos as m

c = m.get_cosmos_container()
items = list(c.query_items(
    'SELECT c.id, c.source_path, c.blob_url, c.page_count, c.status '
    'FROM c WHERE CONTAINS(c.source_path, "08-01")',
    enable_cross_partition_query=True,
))
print("ITEMS:", items)
