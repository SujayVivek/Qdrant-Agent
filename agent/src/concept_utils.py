#!/usr/bin/env python3
"""
Concept Detection & Matching Utilities
Functions for detecting and matching similar concepts in the memory layer.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL


# =====================
# Configuration
# =====================

CONCEPT_COLLECTION = "concept_memory"
SIMILARITY_THRESHOLD = 0.8  # Cosine similarity threshold for concept matching
EMBEDDING_MODEL = "text-embedding-3-small"


# =====================
# Concept Utilities
# =====================

def embed_concept(
    question: str,
    embeddings_model: OpenAIEmbeddings = None
) -> List[float]:
    """
    Convert a user question into a concept embedding.
    
    Args:
        question: The user's question text
        embeddings_model: Optional pre-initialized embeddings model
        
    Returns:
        Embedding vector for the concept
    """
    
    # Initialize embeddings model if not provided
    if embeddings_model is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Generate embedding
    embedding = embeddings_model.embed_query(question)
    return embedding


def find_similar_concept(
    question: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    qdrant_client: QdrantClient = None,
    embeddings_model: OpenAIEmbeddings = None
) -> Optional[Dict]:
    """
    Search for a similar concept in the memory collection.
    
    Args:
        question: The user's question text
        similarity_threshold: Minimum similarity score to consider a match (default: 0.8)
        qdrant_client: Optional pre-initialized Qdrant client
        embeddings_model: Optional pre-initialized embeddings model
        
    Returns:
        Dictionary with concept info if found, None if this is a new concept:
        - concept_id: The ID of the matched concept
        - concept_text: The original concept text
        - similarity: The similarity score (0-1)
        - top_chunks: List of chunk_ids that worked well for this concept
        - usage_count: How many times this concept has been used
        - success_score: Success metric for this concept
        - last_used: When this concept was last accessed
    """
    
    # Initialize Qdrant client if not provided
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    
    # Generate embedding for the question
    query_vector = embed_concept(question, embeddings_model)
    
    # Search for similar concepts
    try:
        search_results = qdrant_client.search(
            collection_name=CONCEPT_COLLECTION,
            query_vector=query_vector,
            limit=1
        )
        
        # Check if we found a match above the threshold
        if search_results and search_results[0].score >= similarity_threshold:
            result = search_results[0]
            
            return {
                "concept_id": result.id,
                "concept_text": result.payload.get("concept_text"),
                "similarity": result.score,
                "top_chunks": result.payload.get("top_chunks", []),
                "usage_count": result.payload.get("usage_count", 1),
                "success_score": result.payload.get("success_score", 0.5),
                "last_used": result.payload.get("last_used")
            }
        else:
            # No similar concept found or below threshold
            return None
            
    except Exception as e:
        # Collection might not exist or be empty
        print(f"⚠️  Warning: Could not search concept memory: {e}")
        return None


def is_new_concept(
    question: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    qdrant_client: QdrantClient = None,
    embeddings_model: OpenAIEmbeddings = None
) -> bool:
    """
    Determine if a question represents a new concept.
    
    Args:
        question: The user's question text
        similarity_threshold: Minimum similarity score to consider a match (default: 0.8)
        qdrant_client: Optional pre-initialized Qdrant client
        embeddings_model: Optional pre-initialized embeddings model
        
    Returns:
        True if this is a new concept, False if similar concept exists
    """
    
    similar_concept = find_similar_concept(
        question,
        similarity_threshold,
        qdrant_client,
        embeddings_model
    )
    
    return similar_concept is None


def update_concept_usage(
    concept_id: str,
    chunks_used: List[str],
    success_feedback: Optional[float] = None,
    qdrant_client: QdrantClient = None
) -> bool:
    """
    Update a concept's usage statistics.
    
    Args:
        concept_id: The ID of the concept to update
        chunks_used: List of chunk_ids that were used for this query
        success_feedback: Optional success score (0-1) for this interaction
        qdrant_client: Optional pre-initialized Qdrant client
        
    Returns:
        True if update was successful, False otherwise
    """
    
    # Initialize Qdrant client if not provided
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    
    try:
        # Get current concept
        concept = qdrant_client.retrieve(
            collection_name=CONCEPT_COLLECTION,
            ids=[concept_id]
        )[0]
        
        # Update payload
        updated_payload = concept.payload.copy()
        updated_payload["usage_count"] = updated_payload.get("usage_count", 0) + 1
        updated_payload["last_used"] = datetime.utcnow().isoformat()
        
        # Update top_chunks (merge with existing)
        existing_chunks = set(updated_payload.get("top_chunks", []))
        new_chunks = set(chunks_used)
        updated_payload["top_chunks"] = list(existing_chunks | new_chunks)
        
        # Update success score if feedback provided
        if success_feedback is not None:
            current_score = updated_payload.get("success_score", 0.5)
            usage_count = updated_payload["usage_count"]
            # Running average of success scores
            updated_payload["success_score"] = (
                (current_score * (usage_count - 1) + success_feedback) / usage_count
            )
        
        # Update in Qdrant
        qdrant_client.set_payload(
            collection_name=CONCEPT_COLLECTION,
            payload=updated_payload,
            points=[concept_id]
        )
        
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Could not update concept: {e}")
        return False


def get_concept_summary(
    qdrant_client: QdrantClient = None
) -> Dict:
    """
    Get a summary of all concepts in memory.
    
    Args:
        qdrant_client: Optional pre-initialized Qdrant client
        
    Returns:
        Dictionary with summary statistics
    """
    
    # Initialize Qdrant client if not provided
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    
    try:
        collection_info = qdrant_client.get_collection(collection_name=CONCEPT_COLLECTION)
        
        return {
            "total_concepts": collection_info.points_count,
            "collection_status": collection_info.status.name,
            "vectors_count": collection_info.vectors_count
        }
    except Exception as e:
        return {
            "total_concepts": 0,
            "error": str(e)
        }


# =====================
# CLI for Testing
# =====================

def main():
    """CLI for testing concept detection"""
    
    if len(sys.argv) < 2:
        print("Usage: python concept_utils.py <question>")
        print('Example: python concept_utils.py "How does Python handle errors?"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print("="*70)
    print("CONCEPT DETECTION")
    print("="*70)
    print(f"\n📝 Question: {question}")
    print(f"🎯 Similarity threshold: {SIMILARITY_THRESHOLD}\n")
    
    try:
        print("🔗 Connecting to Qdrant...")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        print("   ✓ Connected\n")
        
        # Get summary
        summary = get_concept_summary(qdrant_client)
        print(f"📊 Concept Memory Status:")
        print(f"   Total concepts: {summary.get('total_concepts', 0)}")
        print()
        
        # Check if this is a new concept
        print("🔍 Searching for similar concepts...")
        similar_concept = find_similar_concept(question, qdrant_client=qdrant_client)
        
        if similar_concept:
            print("\n✅ FOUND SIMILAR CONCEPT")
            print("="*70)
            print(f"Concept ID: {similar_concept['concept_id']}")
            print(f"Original Text: {similar_concept['concept_text']}")
            print(f"Similarity Score: {similar_concept['similarity']:.4f}")
            print(f"Usage Count: {similar_concept['usage_count']}")
            print(f"Success Score: {similar_concept['success_score']:.3f}")
            print(f"Top Chunks: {len(similar_concept['top_chunks'])} chunks")
            print(f"Last Used: {similar_concept['last_used']}")
            print("="*70)
        else:
            print("\n🆕 NEW CONCEPT DETECTED")
            print("="*70)
            print("No similar concept found in memory.")
            print("This question represents a new learning opportunity.")
            print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
