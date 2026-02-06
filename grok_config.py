"""
Grok (XAI) Configuration Module for LennyHub RAG

This module configures the system to use Grok's API instead of OpenAI.
Grok's API is OpenAI-compatible, so we just need to update the base URL and model names.

Usage:
    from grok_config import get_llm_model_func, get_embedding_func
    
    llm_func = get_llm_model_func()
    embedding_func = get_embedding_func()
"""

import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()


def configure_openai_for_grok():
    """
    Configure the OpenAI client to use Grok's API endpoint.
    This should be called before any OpenAI API calls.
    """
    api_base = os.getenv("OPENAI_API_BASE", "https://api.x.ai/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY (Grok API key) is not set in environment variables!")
    
    # Set the API base URL for OpenAI client
    openai.api_base = api_base
    openai.base_url = api_base
    
    print(f"✓ Configured to use Grok API at: {api_base}")
    return api_base, api_key


def get_llm_model():
    """Get the LLM model name from environment or use default."""
    return os.getenv("LLM_MODEL", "grok-beta")


def get_embedding_model():
    """Get the embedding model name from environment or use default."""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


async def grok_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    LLM function that uses Grok's API.
    This is a wrapper around the standard OpenAI complete function.
    """
    from lightrag.llm.openai import openai_complete_if_cache
    
    # Get model from environment
    model = get_llm_model()
    
    try:
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_base=os.getenv("OPENAI_API_BASE", "https://api.x.ai/v1"),
            **kwargs
        )
    except Exception as e:
        print(f"Error calling Grok API: {e}")
        print(f"Make sure your Grok API key is set in .env file")
        print(f"Get your free Grok API key from: https://console.x.ai/")
        raise


async def grok_embedding_func(texts):
    """
    Embedding function. 
    NOTE: Grok doesn't provide embeddings yet, so we need to use an alternative.
    
    Options:
    1. Use OpenAI embeddings (requires OpenAI API key)
    2. Use local embeddings (sentence-transformers)
    3. Use a different embedding service
    
    For now, we'll try to use the lightrag openai_embed but this will fail
    if you don't have OpenAI API key. You may need to switch to local embeddings.
    """
    from lightrag.llm.openai import openai_embed
    import numpy as np
    
    # Try to use OpenAI embeddings (you can replace this with local embeddings)
    try:
        # This will use OpenAI's embedding endpoint
        # If you want to use Grok only, you'll need to implement local embeddings
        return await openai_embed(texts, model=get_embedding_model())
    except Exception as e:
        print(f"Warning: Embedding failed: {e}")
        print("Note: Grok doesn't provide embeddings yet.")
        print("You have two options:")
        print("1. Set a separate OPENAI_API_KEY for embeddings only")
        print("2. Use local embeddings (not yet implemented in this config)")
        raise


def print_grok_info():
    """Print information about the Grok configuration."""
    print("\n" + "=" * 70)
    print("GROK (XAI) API CONFIGURATION")
    print("=" * 70)
    print(f"API Base URL: {os.getenv('OPENAI_API_BASE', 'https://api.x.ai/v1')}")
    print(f"LLM Model: {get_llm_model()}")
    print(f"Embedding Model: {get_embedding_model()}")
    print(f"API Key Set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    print("\nNote: Grok provides FREE API access!")
    print("Get your API key from: https://console.x.ai/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """Test Grok configuration"""
    print_grok_info()
    
    try:
        configure_openai_for_grok()
        print("✓ Configuration successful!")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
