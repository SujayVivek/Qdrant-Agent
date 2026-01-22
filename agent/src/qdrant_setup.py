#!/usr/bin/env python3
"""
Qdrant Collection Setup
Creates all required collections for the video transcript chatbot system.
Safe to run multiple times - only creates collections that don't exist.
"""

import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL


# Configuration
COLLECTIONS = {
    "video_segments_test": {
        "vector_size": 1536,
        "distance": Distance.COSINE,
        "description": "Video transcript chunks with embeddings"
    },
    "concept_memory": {
        "vector_size": 1536,
        "distance": Distance.COSINE,
        "description": "Learned concepts for improved retrieval"
    }
}


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance
) -> bool:
    """
    Create a collection only if it doesn't already exist.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
        vector_size: Dimension of the vectors
        distance: Distance metric to use
        
    Returns:
        True if collection was created, False if it already existed
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)
    
    if collection_exists:
        print(f"   ℹ️  Collection '{collection_name}' already exists (skipping)")
        return False
    
    # Create collection
    print(f"   🔨 Creating collection: '{collection_name}'")
    print(f"      → Vector size: {vector_size}")
    print(f"      → Distance metric: {distance}")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )
    
    print(f"      ✓ Created successfully")
    return True


def verify_collections(client: QdrantClient) -> None:
    """
    Verify all collections exist and display their info.
    
    Args:
        client: Qdrant client instance
    """
    print("\n" + "="*70)
    print("📊 COLLECTION SUMMARY")
    print("="*70)
    
    for collection_name, config in COLLECTIONS.items():
        try:
            collection_info = client.get_collection(collection_name)
            print(f"\n✓ {collection_name}")
            print(f"  Description: {config['description']}")
            print(f"  Vector size: {collection_info.config.params.vectors.size}")
            print(f"  Distance: {collection_info.config.params.vectors.distance.name}")
            print(f"  Points count: {collection_info.points_count}")
            print(f"  Status: {collection_info.status.name}")
            
        except Exception as e:
            print(f"\n✗ {collection_name}")
            print(f"  Error: {e}")
    
    print("\n" + "="*70)


def main():
    """
    Main entry point - creates all required collections
    """
    print("\n" + "="*70)
    print("🚀 QDRANT COLLECTION SETUP")
    print("="*70)
    print(f"Connecting to: {QDRANT_URL}\n")
    
    try:
        # Connect to Qdrant
        client = QdrantClient(url=QDRANT_URL)
        print("✓ Connected to Qdrant\n")
        
        # Create collections
        print("📦 Setting up collections...")
        print("-"*70)
        
        created_count = 0
        for collection_name, config in COLLECTIONS.items():
            was_created = create_collection_if_not_exists(
                client=client,
                collection_name=collection_name,
                vector_size=config["vector_size"],
                distance=config["distance"]
            )
            if was_created:
                created_count += 1
        
        print("-"*70)
        print(f"\n📈 Results: {created_count} new collection(s) created, "
              f"{len(COLLECTIONS) - created_count} already existed")
        
        # Verify all collections
        verify_collections(client)
        
        print("\n✅ Qdrant setup complete!")
        print("   All collections are ready for use.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Failed to setup Qdrant: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
