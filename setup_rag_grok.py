"""
Setup RAG with Grok (XAI) API + Local Embeddings

This is a modified version of setup_rag.py that uses:
- Grok API for LLM (text generation) - FREE!
- SentenceTransformers for embeddings (local, no API needed) - FREE!

Usage:
    python setup_rag_grok.py --quick                  # Process first 10 transcripts
    python setup_rag_grok.py --max 50                 # Process first 50 transcripts  
    python setup_rag_grok.py --parallel --workers 5   # Parallel processing
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import subprocess
import requests
import time

# Load environment variables
load_dotenv()

# Import the original setup functions
from setup_rag import (
    print_header,
    print_step,
    check_qdrant_installed,
    install_qdrant,
    is_qdrant_running,
    start_qdrant,
    wait_for_qdrant,
    get_already_processed_docs,
    processed_count,
    total_to_process,
    lock,
)


def check_grok_api_key():
    """Check if Grok (XAI) API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.x.ai/v1")
    
    if not api_key or api_key == "your-grok-api-key-here":
        print("ERROR: Grok API key not set!")
        print("\n🔑 Get your FREE Grok API key:")
        print("   1. Visit: https://console.x.ai/")
        print("   2. Sign up/login with your X (Twitter) account")
        print("   3. Create an API key")
        print("   4. Add it to the .env file:")
        print(f"      OPENAI_API_KEY=your-grok-api-key-here")
        print(f"\n   Also make sure this is in .env:")
        print(f"      OPENAI_API_BASE={api_base}")
        return False
    
    print(f"✓ Grok API key found!")
    print(f"  API Base: {api_base}")
    print(f"  Model: {os.getenv('LLM_MODEL', 'grok-beta')}")
    return True


def check_sentence_transformers():
    """Check if sentence-transformers is installed"""
    try:
        import sentence_transformers
        print("✓ sentence-transformers is installed!")
        return True
    except ImportError:
        print("⚠️  sentence-transformers not found. Installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "sentence-transformers", "-q"
            ])
            print("✓ sentence-transformers installed successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Failed to install sentence-transformers: {e}")
            print("\nPlease install manually:")
            print("  pip install sentence-transformers")
            return False


async def build_rag_with_grok(max_transcripts=None, workers=5, use_parallel=False):
    """Build RAG system using Grok for LLM and local embeddings"""
    global total_to_process, processed_count
    
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc
    from qdrant_config import get_lightrag_kwargs
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # Configure API base URL for Grok
    api_base = os.getenv("OPENAI_API_BASE", "https://api.x.ai/v1")
    llm_model = os.getenv("LLM_MODEL", "grok-beta")
    
    print(f"\n🤖 Using Grok LLM: {llm_model}")
    print(f"📊 Using Local Embeddings: sentence-transformers/all-MiniLM-L6-v2")
    print(f"🌐 API Base: {api_base}\n")
    
    start_time = datetime.now()
    
    # Load local embedding model
    print("Loading local embedding model (this may take a minute on first run)...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✓ Embedding model loaded!\n")
    
    # Configure RAG system
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    # Set up Grok LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        """LLM function using Grok API"""
        import openai
        from lightrag.llm.openai import openai_complete_if_cache
        
        # Temporarily set the API base
        original_base = getattr(openai, 'base_url', None)
        openai.base_url = api_base
        
        try:
            result = await openai_complete_if_cache(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
            return result
        finally:
            # Restore original base
            if original_base:
                openai.base_url = original_base
    
    # Set up local embedding function
    async def embedding_func(texts: list[str]) -> np.ndarray:
        """Local embedding function using sentence-transformers"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            embedding_model.encode, 
            texts
        )
        return np.array(embeddings)
    
    # Get Qdrant configuration
    lightrag_kwargs = get_lightrag_kwargs(verbose=False)
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,  # all-MiniLM-L6-v2 produces 384-dim vectors
            max_token_size=256,
            func=embedding_func
        ),
        lightrag_kwargs=lightrag_kwargs
    )
    
    # Get transcript files
    transcript_dir = Path("./data")
    all_files = sorted(list(transcript_dir.glob("*.txt")))
    
    # Get already processed documents
    already_processed = get_already_processed_docs()
    print(f"Already processed: {len(already_processed)} transcripts")
    
    # Filter out already processed
    transcript_files = []
    for file in all_files:
        doc_id = f"transcript-{file.stem}"
        if doc_id not in already_processed:
            transcript_files.append(file)
    
    # Apply max limit
    if max_transcripts and len(transcript_files) > max_transcripts:
        transcript_files = transcript_files[:max_transcripts]
    
    total_to_process = len(transcript_files)
    
    if total_to_process == 0:
        print("\n✓ All transcripts already processed!")
        print(f"Total documents in system: {len(already_processed)}")
        rag.close()
        return True
    
    print(f"\nWill process: {total_to_process} new transcript(s)")
    
    # Ensure LightRAG is initialized
    await rag._ensure_lightrag_initialized()
    
    # Process transcripts
    print("\nProcessing transcripts...\n")
    print("=" * 70 + "\n")
    
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"[{i}/{total_to_process}] Processing: {transcript_file.name}")
        
        # Read transcript content
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert into RAG system
        content_list = [{
            "type": "text",
            "text": content,
            "page_idx": 0
        }]
        
        try:
            await rag.insert_content_list(
                content_list=content_list,
                file_path=str(transcript_file),
                doc_id=f"transcript-{transcript_file.stem}"
            )
            print(f"  ✓ Successfully indexed!\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed: {total_to_process} transcripts")
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    if total_to_process > 0:
        print(f"Average: {duration/total_to_process:.1f} seconds per transcript")
    print(f"Total documents in system: {len(already_processed) + total_to_process}")
    print("=" * 70)
    
    # Test with a sample question
    print("\nTesting RAG system with Grok...\n")
    print("Question: What is a curiosity loop and how does it work?")
    print("-" * 70 + "\n")
    
    try:
        response = await rag.aquery(
            "What is a curiosity loop and how does it work?",
            mode="hybrid"
        )
        print("Answer:")
        print(response)
        print()
    except Exception as e:
        print(f"Warning: Test query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Close RAG system
    rag.close()
    
    return True


async def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup RAG system with Grok API")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Process only first 10 transcripts (for quick testing)"
    )
    parser.add_argument(
        "--max",
        type=int,
        help="Maximum number of transcripts to process"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (currently not implemented with Grok)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers for parallel mode (default: 5)"
    )
    args = parser.parse_args()
    
    # Determine max transcripts
    max_transcripts = None
    if args.quick:
        max_transcripts = 10
    elif args.max:
        max_transcripts = args.max
    
    print_header("🚀 RAG Setup with Grok (XAI) API + Local Embeddings")
    print("This uses:")
    print("  • Grok API for LLM (text generation) - FREE!")
    print("  • Local embeddings (sentence-transformers) - FREE!")
    print("  • Qdrant for vector storage - FREE!")
    print()
    
    # Step 1: Check Grok API key
    print_step(1, "Checking Grok API Configuration")
    if not check_grok_api_key():
        return 1
    
    # Step 2: Check sentence-transformers
    print_step(2, "Checking Local Embedding Library")
    if not check_sentence_transformers():
        return 1
    
    # Step 3: Check/install Qdrant
    print_step(3, "Checking Qdrant Installation")
    if not check_qdrant_installed():
        print("Qdrant not found. Installing...")
        if not install_qdrant():
            print("\n⚠️  Automatic installation failed.")
            print("Please install Qdrant manually:")
            print("  Windows: .\\install_qdrant_windows.ps1")
            print("  Mac/Linux: ./install_qdrant_local.sh")
            return 1
    else:
        print("✓ Qdrant is already installed!")
    
    # Step 4: Start Qdrant
    print_step(4, "Starting Qdrant Server")
    if is_qdrant_running():
        print("✓ Qdrant is already running!")
    else:
        if not start_qdrant():
            return 1
        if not wait_for_qdrant():
            return 1
    
    # Step 5: Build RAG
    print_step(5, "Building RAG System with Grok")
    
    mode_desc = f"first {max_transcripts} transcripts" if max_transcripts else "all transcripts"
    print(f"Processing {mode_desc}...\n")
    
    try:
        success = await build_rag_with_grok(
            max_transcripts=max_transcripts,
            workers=args.workers,
            use_parallel=args.parallel
        )
        
        if not success:
            return 1
    except Exception as e:
        print(f"\nERROR: Failed to build RAG system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print_header("✅ Setup Complete!")
    print("Your RAG system is ready with Grok!\n")
    print("Next steps:")
    print("  1. Run the Streamlit app:")
    print("     streamlit run streamlit_app.py\n")
    print("  2. Or query from command line:")
    print('     python query_rag.py "Your question here"\n')
    print("  3. Check Qdrant dashboard:")
    print("     http://localhost:6333/dashboard\n")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
