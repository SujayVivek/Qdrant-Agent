#!/usr/bin/env python3
"""
Feedback Handler Module
Manages user feedback to update concept memory and improve future retrieval.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL
from concept_utils import find_similar_concept, embed_concept, update_concept_usage
from concept_memory import add_concept


# =====================
# Configuration
# =====================

CONCEPT_COLLECTION = "concept_memory"
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.8


# =====================
# Feedback Handler
# =====================

def handle_feedback(
    user_question: str,
    retrieved_chunks: List[Dict],
    user_feedback: bool,
    qdrant_client: QdrantClient = None,
    embeddings_model: OpenAIEmbeddings = None
) -> Dict:
    """
    Process user feedback to update concept memory.
    
    Args:
        user_question: The original user question
        retrieved_chunks: List of chunk dictionaries that were retrieved
        user_feedback: True if helpful, False if not helpful
        qdrant_client: Optional pre-initialized Qdrant client
        embeddings_model: Optional pre-initialized embeddings model
        
    Returns:
        Dictionary with feedback processing results:
        - action: "created", "updated", or "skipped"
        - concept_id: ID of the concept (if created/updated)
        - message: Human-readable description of what happened
    """
    
    # Initialize clients if not provided
    if embeddings_model is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    
    # If feedback is negative, don't update memory
    if not user_feedback:
        return {
            "action": "skipped",
            "concept_id": None,
            "message": "Negative feedback - memory not updated"
        }
    
    # Extract chunk IDs from retrieved chunks
    chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks if chunk.get("chunk_id")]
    
    if not chunk_ids:
        return {
            "action": "skipped",
            "concept_id": None,
            "message": "No valid chunk IDs to store"
        }
    
    # Use the user question as the concept text (simple and safe)
    concept_text = user_question
    
    # Generate concept embedding
    concept_vector = embed_concept(concept_text, embeddings_model)
    
    # Check if similar concept already exists
    similar_concept = find_similar_concept(
        concept_text,
        similarity_threshold=SIMILARITY_THRESHOLD,
        qdrant_client=qdrant_client,
        embeddings_model=embeddings_model
    )
    
    if similar_concept:
        # Update existing concept
        concept_id = similar_concept["concept_id"]
        
        # Success score: 1.0 for positive feedback
        success_feedback = 1.0
        
        success = update_concept_usage(
            concept_id=concept_id,
            chunks_used=chunk_ids,
            success_feedback=success_feedback,
            qdrant_client=qdrant_client
        )
        
        if success:
            return {
                "action": "updated",
                "concept_id": concept_id,
                "message": f"Updated existing concept (usage count increased, chunks reinforced)",
                "previous_usage": similar_concept["usage_count"],
                "similarity": similar_concept["similarity"]
            }
        else:
            return {
                "action": "error",
                "concept_id": concept_id,
                "message": "Failed to update concept"
            }
    
    else:
        # Create new concept
        import uuid
        concept_id = str(uuid.uuid4())
        
        try:
            add_concept(
                concept_text=concept_text,
                concept_vector=concept_vector,
                top_chunks=chunk_ids,
                qdrant_client=qdrant_client,
                concept_id=concept_id
            )
            
            return {
                "action": "created",
                "concept_id": concept_id,
                "message": f"Created new concept with {len(chunk_ids)} helpful chunks",
                "chunks_stored": len(chunk_ids)
            }
        
        except Exception as e:
            return {
                "action": "error",
                "concept_id": None,
                "message": f"Failed to create concept: {str(e)}"
            }


def process_positive_feedback(
    user_question: str,
    retrieved_chunks: List[Dict],
    qdrant_client: QdrantClient = None,
    embeddings_model: OpenAIEmbeddings = None
) -> Dict:
    """
    Convenience function for processing positive feedback.
    
    Args:
        user_question: The original user question
        retrieved_chunks: List of chunk dictionaries that were retrieved
        qdrant_client: Optional pre-initialized Qdrant client
        embeddings_model: Optional pre-initialized embeddings model
        
    Returns:
        Dictionary with feedback processing results
    """
    return handle_feedback(
        user_question,
        retrieved_chunks,
        user_feedback=True,
        qdrant_client=qdrant_client,
        embeddings_model=embeddings_model
    )


def process_negative_feedback(
    user_question: str,
    retrieved_chunks: List[Dict]
) -> Dict:
    """
    Convenience function for processing negative feedback.
    Currently does not update memory.
    
    Args:
        user_question: The original user question
        retrieved_chunks: List of chunk dictionaries that were retrieved
        
    Returns:
        Dictionary indicating no action was taken
    """
    return {
        "action": "skipped",
        "concept_id": None,
        "message": "Negative feedback - memory not updated"
    }


# =====================
# CLI for Testing
# =====================

def main():
    """CLI for testing feedback handler"""
    
    print("="*70)
    print("FEEDBACK HANDLER TEST")
    print("="*70)
    print()
    
    # Sample test data
    test_question = "How does Python handle exceptions?"
    test_chunks = [
        {
            "chunk_id": "test_chunk_1",
            "video_id": "test_video",
            "start_time": 10.5,
            "end_time": 25.3,
            "text": "Python uses try-except blocks...",
            "score": 0.95
        },
        {
            "chunk_id": "test_chunk_2",
            "video_id": "test_video",
            "start_time": 30.1,
            "end_time": 45.7,
            "text": "The finally clause always executes...",
            "score": 0.87
        }
    ]
    
    print(f"📝 Test Question: {test_question}")
    print(f"📊 Test Chunks: {len(test_chunks)} chunks")
    print()
    
    # Test positive feedback
    print("Testing POSITIVE feedback...")
    try:
        result = process_positive_feedback(test_question, test_chunks)
        
        print(f"\n✅ Result:")
        print(f"   Action: {result['action']}")
        print(f"   Message: {result['message']}")
        if result.get('concept_id'):
            print(f"   Concept ID: {result['concept_id']}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test negative feedback
    print("="*70)
    print("Testing NEGATIVE feedback...")
    result = process_negative_feedback(test_question, test_chunks)
    print(f"\n✅ Result:")
    print(f"   Action: {result['action']}")
    print(f"   Message: {result['message']}")
    print()
    
    print("="*70)
    print("Feedback handler test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
