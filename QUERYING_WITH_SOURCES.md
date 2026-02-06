# Querying RAG with Chunk-Level Source Attribution

This guide explains how to query the RAG system and see the **specific chunks** used to generate answers, rather than just the overall source document.

## Available Query Scripts

### 1. `query_rag.py` (Original)
**Shows:** Final answer only

```bash
python query_rag.py "What is a curiosity loop?"
python query_rag.py --interactive
```

**Output:**
- ✅ Generated answer
- ❌ No source information
- ❌ No chunk details

---

### 2. `query_rag_with_chunks.py` (NEW)
**Shows:** Retrieved chunks + Final answer

```bash
python query_rag_with_chunks.py "What is a curiosity loop?"
python query_rag_with_chunks.py --interactive
```

**Output:**
```
📚 RETRIEVED CHUNKS (Used to generate the answer):
================================================================================

Found 5 relevant chunks:

────────────────────────────────────────────────────────────────────────────────
CHUNK 1:
────────────────────────────────────────────────────────────────────────────────
[Actual text content from the knowledge base that was used...]

────────────────────────────────────────────────────────────────────────────────
CHUNK 2:
────────────────────────────────────────────────────────────────────────────────
[More text content...]

💡 GENERATED ANSWER:
================================================================================
[Final synthesized answer based on the chunks above]
```

**Interactive Mode Commands:**
- `show-chunks` - Enable chunk display
- `no-chunks` - Disable chunk display
- `quit` - Exit

---

### 3. `query_with_sources.py` (NEW - Most Detailed)
**Shows:** Retrieved chunks + Source documents + Final answer

```bash
python query_with_sources.py "What is a curiosity loop?"
```

**Output:**
```
📚 RETRIEVED SOURCES:
====================================================================================================

Retrieved from 2 source document(s):

════════════════════════════════════════════════════════════════════════════════════════════════════
📄 SOURCE: Ada Chen Rekhi
   Chunks used: 3
════════════════════════════════════════════════════════════════════════════════════════════════════

┌─ Chunk 1 ───────────────────────────────────────────────────────────────────────────────────────
│ Chunk ID: chunk-3ef01b14864e33f95632c541afb0e371
│
│ [Specific text from Ada Chen Rekhi transcript...]
└──────────────────────────────────────────────────────────────────────────────────────────────────

┌─ Chunk 2 ───────────────────────────────────────────────────────────────────────────────────────
│ Chunk ID: chunk-813fa4d6ddad5e5e9b3c0409a3eb009e
│
│ [More text from Ada Chen Rekhi transcript...]
└──────────────────────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════════════════════════
📄 SOURCE: Adam Fishman
   Chunks used: 2
════════════════════════════════════════════════════════════════════════════════════════════════════

[Additional chunks from different source...]

💡 GENERATED ANSWER:
====================================================================================================
[Final answer with citation traceability]

✓ Answer generated using 5 chunk(s) from 2 source(s)
```

---

## How It Works

### The `only_need_context` Parameter

LightRAG provides a parameter called `only_need_context` that:
- ✅ Performs the full retrieval process (vector search + knowledge graph traversal)
- ✅ Returns the retrieved chunks
- ❌ Skips LLM answer generation

**Example:**
```python
from lightrag import QueryParam

# Get chunks without generating answer
query_param = QueryParam(mode="hybrid", only_need_context=True)
context = await rag.lightrag.aquery(question, param=query_param)

# context contains the retrieved chunks as a string
```

### Chunk Metadata

Each chunk is stored with metadata including:
- **Chunk ID**: Unique identifier (e.g., `chunk-3ef01b14864e33f95632c541afb0e371`)
- **Content**: The actual text segment
- **Source Document**: Which transcript/file it came from
- **Embeddings**: Vector representation for similarity search

This metadata is stored in:
- `rag_storage/kv_store_text_chunks.json` - Chunk content and metadata
- `rag_storage/kv_store_doc_status.json` - Document-to-chunk mappings
- `rag_storage/vdb_chunks.json` - Vector embeddings

---

## Comparison Table

| Feature | query_rag.py | query_rag_with_chunks.py | query_with_sources.py |
|---------|--------------|--------------------------|----------------------|
| Final Answer | ✅ | ✅ | ✅ |
| Retrieved Chunks | ❌ | ✅ | ✅ |
| Source Document Names | ❌ | ❌ | ✅ |
| Chunk IDs | ❌ | ❌ | ✅ |
| Grouped by Source | ❌ | ❌ | ✅ |
| Interactive Mode | ✅ | ✅ | ❌ |

---

## Use Cases

### When to Use `query_rag.py`
- Quick queries where you just need the answer
- Production use cases where users don't need to see sources
- When minimizing output length

### When to Use `query_rag_with_chunks.py`
- Debugging: Understanding what context was retrieved
- Research: Seeing the raw text that informed the answer
- Transparency: Showing users what information was used

### When to Use `query_with_sources.py`
- Maximum transparency: Need to trace back to exact source documents
- Multi-document scenarios: Want to see which transcripts contributed
- Citation requirements: Need proper attribution with chunk IDs
- Quality assurance: Verifying retrieval quality

---

## Example Queries

### Question 1: Single-source question
```bash
python query_with_sources.py "What is a curiosity loop?"
```
Expected: Chunks primarily from Ada Chen Rekhi transcript

### Question 2: Multi-source question
```bash
python query_with_sources.py "What are frameworks for career development?"
```
Expected: Chunks from multiple transcripts (Ada Chen Rekhi, Adam Fishman, etc.)

### Question 3: Specific person question
```bash
python query_with_sources.py "What does Adam Fishman say about onboarding?"
```
Expected: Chunks specifically from Adam Fishman transcript

---

## Technical Details

### Retrieval Process

1. **Query Embedding**: Question is converted to vector representation
2. **Entity Search**: Find relevant entities in knowledge graph
3. **Relationship Traversal**: Follow entity connections
4. **Chunk Retrieval**: Get text chunks associated with entities
5. **Ranking**: Sort chunks by relevance
6. **Context Assembly**: Combine chunks with separators
7. **Answer Generation**: LLM synthesizes final answer from context

### Context Format

LightRAG returns context with chunks separated by `"-----"`:
```
[Chunk 1 content]
-----
[Chunk 2 content]
-----
[Chunk 3 content]
```

The scripts parse this format and enhance it with source attribution.

### Chunk Size

Default chunk settings:
- **Max tokens per chunk**: 4000 tokens
- **Overlap**: Configurable (helps maintain context across boundaries)
- **Separator**: `"-----"` between chunks

---

## Benefits of Chunk-Level Attribution

1. **Transparency**: Users see exactly what text informed the answer
2. **Verification**: Easy to check if retrieval is working correctly
3. **Trust**: Builds confidence by showing sources
4. **Debugging**: Identify when wrong chunks are retrieved
5. **Citation**: Proper attribution to original sources
6. **Quality Control**: Spot-check chunk relevance

---

## Cost Implications

### `only_need_context=True` (Chunks Only)
- **Embedding cost**: ~$0.0001 per query
- **No LLM cost**: Skips answer generation
- **Total**: ~$0.0001 per query

### Standard Query (Chunks + Answer)
- **Embedding cost**: ~$0.0001
- **LLM cost**: ~$0.001-0.01 (depending on context size)
- **Total**: ~$0.001-0.01 per query

The new scripts call the API twice (once for chunks, once for answer), so cost is slightly higher but still minimal (~$0.002-0.02 per query).

---

## Next Steps

1. **Try the scripts**: Run example queries with each script
2. **Compare outputs**: See the difference in transparency
3. **Choose the right tool**: Pick the script that fits your needs
4. **Integrate**: Use `only_need_context` in your own applications

---

## References

- **LightRAG Documentation**: https://github.com/HKUDS/LightRAG
- **QueryParam Class**: Includes `only_need_context`, `top_k`, `mode`, etc.
- **RAG-Anything**: https://github.com/HKUDS/RAG-Anything

---

**Created**: January 2026
**Last Updated**: January 2026
