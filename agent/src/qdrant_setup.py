#!/usr/bin/env python3
"""
Qdrant Collection Setup
Creates and configures the video_segments collection for storing embeddings
"""

import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL


# Configuration
COLLECTION_NAME = "video_segments_test"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embedding dimensions
DISTANCE_METRIC = Distance.COSINE


def create_collection(client: QdrantClient, recreate: bool = True) -> None:
    """
    Create or recreate the video_segments collection
    
    Args:
        client: Qdrant client instance
        recreate: If True, delete existing collection before creating
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == COLLECTION_NAME for col in collections)
    
    if collection_exists:
        if recreate:
            print(f"⚠️  Collection '{COLLECTION_NAME}' already exists")
            print(f"   Deleting existing collection...")
            client.delete_collection(COLLECTION_NAME)
            print(f"   ✓ Deleted")
        else:
            print(f"✓ Collection '{COLLECTION_NAME}' already exists (skipping creation)")
            return
    
    # Create collection
    print(f"📦 Creating collection: {COLLECTION_NAME}")
    print(f"   → Vector size: {VECTOR_SIZE}")
    print(f"   → Distance metric: {DISTANCE_METRIC}")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=DISTANCE_METRIC
        )
    )
    
    print(f"   ✓ Collection created successfully")


def verify_collection(client: QdrantClient) -> None:
    """
    Verify collection exists and display info
    
    Args:
        client: Qdrant client instance
    """
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\n📊 Collection Info:")
        print(f"   Name: {collection_info.config.params.vectors.size}")
        print(f"   Vector size: {collection_info.config.params.vectors.size}")
        print(f"   Distance: {collection_info.config.params.vectors.distance}")
        print(f"   Points count: {collection_info.points_count}")
    except Exception as e:
        print(f"   ✗ Failed to verify collection: {e}")


def main():
    """
    Main entry point
    """
    print("🚀 Setting up Qdrant collection...")
    print(f"   Connecting to: {QDRANT_URL}\n")
    
    try:
        # Connect to Qdrant
        client = QdrantClient(url=QDRANT_URL)
        print("✓ Connected to Qdrant\n")
        
        # Create collection
        recreate = True  # Set to False to preserve existing collection
        create_collection(client, recreate=recreate)
        
        # Verify collection
        verify_collection(client)
        
        print(f"\n✅ Qdrant setup complete!")
        print(f"   Collection '{COLLECTION_NAME}' is ready for embeddings")
        
    except Exception as e:
        print(f"\n❌ Failed to setup Qdrant: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
