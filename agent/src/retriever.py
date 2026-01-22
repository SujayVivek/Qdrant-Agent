#!/usr/bin/env python3
"""
Qdrant Retriever Module
Retrieves relevant video segments from Qdrant using semantic search.
Enhanced with concept memory for improved retrieval over time.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL
from concept_utils import find_similar_concept


# =====================
# Configuration
# =====================

COLLECTION_NAME = "video_segments_test"
CONCEPT_COLLECTION = "concept_memory"
EMBEDDING_MODEL = "text-embedding-3-small"
CONCEPT_SIMILARITY_THRESHOLD = 0.8
MEMORY_BOOST_FACTOR = 0.15  # Soft boost for concept memory chunks


# =====================
# Retriever Function
# =====================

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    qdrant_client: QdrantClient = None,
    embeddings_model: OpenAIEmbeddings = None,
    use_concept_memory: bool = True
) -> List[Dict]:
    """
    Retrieve the most relevant video segments for a given query.
    Enhanced with concept memory for improved retrieval over time.
    
    Args:
        query: Natural language query string
        top_k: Number of results to return (default: 5)
        qdrant_client: Optional pre-initialized Qdrant client
        embeddings_model: Optional pre-initialized embeddings model
        use_concept_memory: Whether to use concept memory for boosting (default: True)
        
    Returns:
        List of dictionaries containing:
        - chunk_id: Unique chunk identifier
        - video_id: Source video ID
        - start_time: Start timestamp
        - end_time: End timestamp
        - text: Transcript text
        - score: Similarity score (boosted if from concept memory)
    """
    
    # Initialize embeddings model if not provided
    if embeddings_model is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Initialize Qdrant client if not provided
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    
    # Step 1: Check concept memory for similar concepts
    memory_chunk_ids: Set[str] = set()
    
    if use_concept_memory:
        try:
            similar_concept = find_similar_concept(
                query,
                similarity_threshold=CONCEPT_SIMILARITY_THRESHOLD,
                qdrant_client=qdrant_client,
                embeddings_model=embeddings_model
            )
            
            if similar_concept:
                memory_chunk_ids = set(similar_concept.get("top_chunks", []))
                if memory_chunk_ids:
                    print(f"   💡 Found similar concept with {len(memory_chunk_ids)} known helpful chunks")
        except Exception as e:
            # Concept memory might not exist yet - continue without it
            print(f"   ℹ️  Concept memory not available: {e}")
    
    # Step 2: Perform normal semantic search
    query_vector = embeddings_model.embed_query(query)
    
    # Retrieve more results than needed to allow for re-ranking
    search_limit = top_k * 2 if memory_chunk_ids else top_k
    
    raw_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=search_limit,
        with_payload=True
    )
    
    # Step 3: Process and potentially boost results
    results = []
    for result in raw_results.points:
        chunk_id = result.payload.get("chunk_id")
        score = result.score
        
        # Apply soft boost if chunk is in concept memory
        if chunk_id in memory_chunk_ids:
            score = min(score + MEMORY_BOOST_FACTOR, 1.0)  # Cap at 1.0
        
        results.append({
            "chunk_id": chunk_id,
            "video_id": result.payload.get("video_id"),
            "start_time": result.payload.get("start_time"),
            "end_time": result.payload.get("end_time"),
            "text": result.payload.get("text"),
            "score": score,
            "boosted": chunk_id in memory_chunk_ids
        })
    
    # Step 4: Re-rank by (potentially boosted) score and take top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]
    
    return results


# =====================
# CLI for Testing
# =====================

def main():
    """CLI for testing the retriever"""
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query>")
        print('Example: python retriever.py "How do I handle errors in Python?"')
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print(f"🔍 Query: {query}")
    print(f"🔗 Connecting to Qdrant: {QDRANT_URL}")
    print(f"📚 Collection: {COLLECTION_NAME}\n")
    
    try:
        results = retrieve_relevant_chunks(query, top_k=3)
        
        print(f"✅ Found {len(results)} relevant chunks:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'='*60}")
            print(f"Result {i} (Score: {result['score']:.4f})")
            print(f"{'='*60}")
            print(f"Video ID: {result['video_id']}")
            print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"\nText:\n{result['text'][:200]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
