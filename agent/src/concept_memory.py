#!/usr/bin/env python3
"""
Concept Memory Collection Setup
Creates and manages the "concept_memory" collection in Qdrant for persistent learning.
"""

import sys
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL


# =====================
# Configuration
# =====================

COLLECTION_NAME = "concept_memory"
VECTOR_DIMENSION = 1536  # Matches OpenAI text-embedding-3-small
DISTANCE_METRIC = Distance.COSINE


# =====================
# Collection Setup
# =====================

def create_concept_memory_collection(
    qdrant_client: QdrantClient,
    recreate: bool = False
) -> None:
    """
    Ensure the concept_memory collection exists in Qdrant.
    This function is idempotent.

    Args:
        qdrant_client: Initialized Qdrant client
        recreate: If True, delete and recreate the collection
    """

    collections = qdrant_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if exists and recreate:
        print(f"⚠️  Collection '{COLLECTION_NAME}' exists. Recreating...")
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print("   ✓ Deleted existing collection")
        exists = False

    if not exists:
        print(f"🔨 Creating collection: {COLLECTION_NAME}")
        print(f"   Vector dimension: {VECTOR_DIMENSION}")
        print(f"   Distance metric: {DISTANCE_METRIC}")

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=DISTANCE_METRIC
            )
        )

        print("   ✓ Collection created successfully")
    else:
        print(f"ℹ️  Collection '{COLLECTION_NAME}' already exists. No action needed.")


def get_collection_info(qdrant_client: QdrantClient) -> dict:
    """
    Get information about the concept_memory collection.
    """

    try:
        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "exists": True,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.name
            }
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }


def add_concept(
    concept_text: str,
    concept_vector: list,
    top_chunks: list,
    qdrant_client: QdrantClient,
    concept_id: Optional[str] = None
) -> str:
    """
    Add a new concept to the concept_memory collection.
    """

    if concept_id is None:
        concept_id = str(uuid.uuid4())

    point = PointStruct(
        id=concept_id,
        vector=concept_vector,
        payload={
            "concept_text": concept_text,
            "top_chunks": top_chunks,
            "usage_count": 1,
            "success_score": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat(),
        }
    )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )

    return concept_id


# =====================
# Main
# =====================

def main():
    print("=" * 70)
    print("CONCEPT MEMORY COLLECTION SETUP")
    print("=" * 70)
    print()

    try:
        print(f"🔗 Connecting to Qdrant: {QDRANT_URL}")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        print("   ✓ Connected\n")

        info = get_collection_info(qdrant_client)

        if info["exists"]:
            print("📊 Existing collection status:")
            print(f"   Vector size: {info['config']['vector_size']}")
            print(f"   Distance metric: {info['config']['distance']}")
            print(f"   Points count: {info['points_count']}\n")

            response = input("❓ Recreate collection? (yes/no): ").strip().lower()
            recreate = response in ("yes", "y")
            print()
        else:
            recreate = False

        create_concept_memory_collection(
            qdrant_client=qdrant_client,
            recreate=recreate
        )

        final_info = get_collection_info(qdrant_client)

        if not final_info["exists"]:
            print("❌ Failed to verify concept_memory collection")
            sys.exit(1)

        print("=" * 70)
        print("✅ CONCEPT MEMORY COLLECTION READY")
        print("=" * 70)
        print(f"Collection name : {COLLECTION_NAME}")
        print(f"Vector dimension: {final_info['config']['vector_size']}")
        print(f"Distance metric : {final_info['config']['distance']}")
        print(f"Points count   : {final_info['points_count']}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
