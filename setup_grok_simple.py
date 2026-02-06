"""
Simple Grok RAG Setup - Windows Friendly
No bash dependencies, pure Python
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup RAG with Grok")
    parser.add_argument("--quick", action="store_true", help="Process 10 transcripts")
    parser.add_argument("--max", type=int, help="Max transcripts to process")
    args = parser.parse_args()
    
    max_transcripts = 10 if args.quick else (args.max if args.max else None)
    
    print("\n" + "="*70)
    print("RAG Setup with Grok API + Local Embeddings")
    print("="*70 + "\n")
    
    # Check API key
    print("[1/5] Checking Grok API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-grok-api-key-here":
        print("ERROR: Grok API key not set in .env file!")
        return 1
    print(f"OK - API key found")
    print(f"API Base: {os.getenv('OPENAI_API_BASE', 'https://api.x.ai/v1')}")
    print(f"Model: {os.getenv('LLM_MODEL', 'grok-beta')}\n")
    
    # Check Qdrant
    print("[2/5] Checking Qdrant...")
    import requests
    try:
        r = requests.get("http://localhost:6333/", timeout=2)
        print(f"OK - Qdrant is running (version {r.json().get('version', 'unknown')})\n")
    except:
        print("ERROR: Qdrant is not running!")
        print("Please start Qdrant first:")
        print(f"  cd {os.path.join(os.path.expanduser('~'), '.qdrant')}")
        print("  .\\qdrant.exe --config-path <path-to-project>\\qdrant_config.yaml")
        return 1
    
    # Load sentence-transformers
    print("[3/5] Loading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("OK - Embedding model loaded\n")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        print("Install with: pip install sentence-transformers")
        return 1
    
    # Initialize RAG
    print("[4/5] Initializing RAG system...")
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc
    from qdrant_config import get_lightrag_kwargs
    import numpy as np
    import openai
    
    # Configure
    api_base = os.getenv("OPENAI_API_BASE", "https://api.x.ai/v1")
    llm_model = os.getenv("LLM_MODEL", "grok-beta")
    
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    # LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        from lightrag.llm.openai import openai_complete_if_cache
        original_base = getattr(openai, 'base_url', None)
        openai.base_url = api_base
        try:
            return await openai_complete_if_cache(
                llm_model, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
        finally:
            if original_base:
                openai.base_url = original_base
    
    # Embedding function
    async def embedding_func(texts):
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, embedding_model.encode, texts)
        return np.array(embeddings)
    
    # Initialize
    lightrag_kwargs = get_lightrag_kwargs(verbose=False)
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=256,
            func=embedding_func
        ),
        lightrag_kwargs=lightrag_kwargs
    )
    
    print("OK - RAG system initialized\n")
    
    # Get transcripts to process
    print(f"[5/5] Processing transcripts...")
    transcript_dir = Path("./data")
    all_files = sorted(list(transcript_dir.glob("*.txt")))
    
    # Check already processed
    import json
    doc_status_file = Path("./rag_storage/kv_store_full_docs.json")
    already_processed = set()
    if doc_status_file.exists():
        try:
            with open(doc_status_file, 'r') as f:
                docs = json.load(f)
                already_processed = set(docs.keys())
        except:
            pass
    
    # Filter
    transcript_files = []
    for file in all_files:
        doc_id = f"transcript-{file.stem}"
        if doc_id not in already_processed:
            transcript_files.append(file)
    
    if max_transcripts and len(transcript_files) > max_transcripts:
        transcript_files = transcript_files[:max_transcripts]
    
    if len(transcript_files) == 0:
        print(f"\nAll transcripts already processed! Total: {len(already_processed)}")
        rag.close()
        return 0
    
    print(f"Will process {len(transcript_files)} new transcripts")
    print(f"Already processed: {len(already_processed)}")
    print()
    
    # Initialize LightRAG
    await rag._ensure_lightrag_initialized()
    
    # Process
    start_time = datetime.now()
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"[{i}/{len(transcript_files)}] {transcript_file.name}...", end=" ", flush=True)
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            await rag.insert_content_list(
                content_list=[{
                    "type": "text",
                    "text": content,
                    "page_idx": 0
                }],
                file_path=str(transcript_file),
                doc_id=f"transcript-{transcript_file.stem}"
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print(f"COMPLETE! Processed {len(transcript_files)} transcripts in {duration:.1f}s")
    print(f"Total in system: {len(already_processed) + len(transcript_files)}")
    print("="*70 + "\n")
    
    # Test query
    print("Testing with sample query...")
    try:
        response = await rag.aquery(
            "What is a curiosity loop?",
            mode="hybrid"
        )
        print("\nQuestion: What is a curiosity loop?")
        print("-"*70)
        print(response[:500] + "..." if len(response) > 500 else response)
        print()
    except Exception as e:
        print(f"Warning: Test query failed: {e}\n")
    
    rag.close()
    
    print("\nSUCCESS! You can now run:")
    print("  streamlit run streamlit_app.py\n")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
