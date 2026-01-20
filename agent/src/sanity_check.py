#!/usr/bin/env python3
"""
Sanity check script to verify environment setup and initialize critical integrations
"""

import sys

def main():
    print("🔍 Running sanity checks...")
    print()
    
    # Check 1: Import config and verify environment variables
    print("1. Checking configuration...")
    try:
        from config import ANTHROPIC_API_KEY, HF_API_KEY, QDRANT_URL
        print(f"   ✓ ANTHROPIC_API_KEY loaded: {ANTHROPIC_API_KEY[:8]}...****")
        print(f"   ✓ HF_API_KEY loaded: {HF_API_KEY[:8]}...****")
        print(f"   ✓ QDRANT_URL loaded: {QDRANT_URL}")
    except Exception as e:
        print(f"   ✗ Failed to load config: {e}")
        sys.exit(1)
    
    print()
    
    # Check 2: Initialize LangChain Anthropic chat model
    print("2. Initializing LangChain Anthropic chat model...")
    try:
        from langchain_anthropic import ChatAnthropic
        chat_model = ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model="claude-3-5-sonnet-20241022"
        )
        print(f"   ✓ ChatAnthropic initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize ChatAnthropic: {e}")
        sys.exit(1)
    
    print()
    
    # Check 3: Initialize HuggingFaceEndpointEmbeddings
    print("3. Initializing HuggingFaceEndpointEmbeddings...")
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HF_API_KEY
        )
        print("   ✓ HuggingFaceEndpointEmbeddings initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize HuggingFaceEndpointEmbeddings: {e}")
        sys.exit(1)

    
    print()
    
    # Check 4: Initialize Qdrant client
    print("4. Initializing Qdrant client...")
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=QDRANT_URL)
        print(f"   ✓ Qdrant client initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize Qdrant client: {e}")
        sys.exit(1)
    
    print()
    print("✅ All sanity checks passed!")
    print("🚀 Environment is ready for development")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
