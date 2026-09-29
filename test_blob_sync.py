"""End-to-end test: process one blob and verify it lands in Cosmos DB."""
import logging
logging.basicConfig(level=logging.INFO)
import blobFormRecogniserCosmos as m

# Process a single known PDF blob
item = m.process_blob('csl-source/08-01.pdf', 'csl14')
print("PROCESSED OK")
print("  id:", item['id'])
print("  source_path:", item['source_path'])
print("  blob_url:", item['blob_url'])
print("  pages:", item['page_count'])
print("  chars:", len(item['content']))
print("  status:", item['status'])

# Verify it's in Cosmos
c = m.get_cosmos_container()
items = list(c.query_items(
    'SELECT c.id, c.source_path, c.blob_url, c.page_count, c.status '
    'FROM c WHERE CONTAINS(c.source_path, "08-01.pdf")',
    enable_cross_partition_query=True,
))
print("COSMOS QUERY:", items)
print("TEST PASSED" if items else "TEST FAILED")
