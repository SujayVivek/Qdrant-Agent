#!/usr/bin/env python3
"""
RAG Chatbot Agent
Combines retrieval from Qdrant with Claude for question answering.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from qdrant_client import QdrantClient

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANTHROPIC_API_KEY, QDRANT_URL
from retriever import retrieve_relevant_chunks
from agent_prompt import get_rag_system_prompt, format_context_for_llm


# =====================
# Configuration
# =====================

CLAUDE_MODEL = "claude-3-haiku-20240307"
DEFAULT_TOP_K = 5


# =====================
# Chatbot Agent
# =====================

class VideoChatbotAgent:
    """
    RAG-based chatbot agent for answering questions about video transcripts.
    
    Combines:
    - Semantic retrieval from Qdrant
    - Claude for natural language generation
    """
    
    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        claude_model: str = CLAUDE_MODEL
    ):
        """
        Initialize the chatbot agent.
        
        Args:
            top_k: Number of video segments to retrieve
            claude_model: Claude model identifier
        """
        self.top_k = top_k
        self.claude_model = claude_model
        
        # Initialize components
        print("🤖 Initializing Video Chatbot Agent...")
        
        # OpenAI embeddings for retrieval
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        print("   ✓ OpenAI embeddings initialized")
        
        # Qdrant client
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        print(f"   ✓ Connected to Qdrant: {QDRANT_URL}")
        
        # Claude chat model
        self.llm = ChatAnthropic(
            model=claude_model,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.0,  # Deterministic for factual answers
            max_tokens=2000
        )
        print(f"   ✓ Claude model initialized: {claude_model}")
        
        # System prompt
        self.system_prompt = get_rag_system_prompt()
        print("   ✓ RAG system prompt loaded\n")
    
    def answer_question(self, user_question: str) -> Tuple[str, List[Dict]]:
        """
        Answer a user question using RAG.
        
        Args:
            user_question: The user's natural language question
            
        Returns:
            Tuple of (answer, retrieved_chunks):
            - answer: The LLM's response
            - retrieved_chunks: List of chunks used for context
        """
        
        # Step 1: Retrieve relevant chunks
        print(f"🔍 Retrieving relevant video segments...")
        retrieved_chunks = retrieve_relevant_chunks(
            query=user_question,
            top_k=self.top_k,
            qdrant_client=self.qdrant_client,
            embeddings_model=self.embeddings_model
        )
        print(f"   ✓ Retrieved {len(retrieved_chunks)} segments\n")
        
        # Step 2: Format context
        context = format_context_for_llm(retrieved_chunks)
        
        # Step 3: Build messages for Claude
        messages = [
            {"role": "user", "content": f"{context}\n\n---\n\nUser Question: {user_question}"}
        ]
        
        # Step 4: Get response from Claude
        print(f"🧠 Generating answer with Claude...")
        response = self.llm.invoke(
            messages,
            system=self.system_prompt
        )
        
        answer = response.content
        print(f"   ✓ Answer generated\n")
        
        return answer, retrieved_chunks
    
    def chat(self, user_question: str, verbose: bool = True) -> str:
        """
        Convenience method for getting just the answer.
        
        Args:
            user_question: The user's question
            verbose: Whether to print retrieved chunks
            
        Returns:
            The agent's answer
        """
        answer, chunks = self.answer_question(user_question)
        
        if verbose:
            print("="*70)
            print("RETRIEVED CONTEXT")
            print("="*70)
            for i, chunk in enumerate(chunks, 1):
                print(f"\n{i}. Video: {chunk['video_id']}, "
                      f"Time: {chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s "
                      f"(Score: {chunk['score']:.3f})")
                print(f"   {chunk['text'][:150]}...")
            print()
        
        return answer


# =====================
# Standalone Function
# =====================

def answer_question(user_question: str, top_k: int = DEFAULT_TOP_K) -> Dict:
    """
    Standalone function to answer a question using RAG.
    
    Args:
        user_question: The user's natural language question
        top_k: Number of video segments to retrieve
        
    Returns:
        Dictionary containing:
        - answer: The LLM's response
        - retrieved_chunks: Context used for reasoning
        - metadata: Additional info about the response
    """
    
    # Initialize agent
    agent = VideoChatbotAgent(top_k=top_k)
    
    # Get answer
    answer, retrieved_chunks = agent.answer_question(user_question)
    
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "metadata": {
            "model": CLAUDE_MODEL,
            "num_chunks_retrieved": len(retrieved_chunks),
            "top_k": top_k
        }
    }


# =====================
# CLI for Testing
# =====================

def main():
    """CLI interface for testing the chatbot"""
    
    if len(sys.argv) < 2:
        print("Usage: python chatbot_agent.py <question>")
        print('Example: python chatbot_agent.py "How does error handling work?"')
        sys.exit(1)
    
    user_question = " ".join(sys.argv[1:])
    
    print("="*70)
    print("VIDEO TRANSCRIPT CHATBOT")
    print("="*70)
    print(f"\n📝 Question: {user_question}\n")
    print("="*70)
    print()
    
    try:
        # Initialize agent
        agent = VideoChatbotAgent(top_k=5)
        
        # Get answer
        answer, chunks = agent.answer_question(user_question)
        
        # Display results
        print("="*70)
        print("RETRIEVED CONTEXT")
        print("="*70)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{i}. Video: {chunk['video_id']}")
            print(f"   Time: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
            print(f"   Score: {chunk['score']:.4f}")
            print(f"   Text: {chunk['text'][:200]}...")
        
        print("\n" + "="*70)
        print("AGENT ANSWER")
        print("="*70)
        print(f"\n{answer}\n")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
