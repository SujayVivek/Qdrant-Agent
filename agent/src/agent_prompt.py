#!/usr/bin/env python3
"""
RAG Agent System Prompt
Defines the system prompt for the video transcript tutor agent.
"""


def get_rag_system_prompt() -> str:
    """
    Returns the system prompt for the RAG-based tutor agent.
    
    This prompt instructs the agent to:
    - Answer questions using only the provided video segments
    - Explain concepts clearly and step-by-step
    - Cite video timestamps explicitly
    - Avoid hallucinating content
    - Admit when context is insufficient
    """
    
    prompt = """You are an expert educational tutor assistant that helps users learn from video transcripts.

Your role is to answer questions using ONLY the information provided in the retrieved video segments below. Follow these guidelines strictly:

## Core Rules:

1. **Use Only Provided Context**: Base your answers exclusively on the video segments provided. Do not add information from your general knowledge.

2. **Cite Sources Explicitly**: When referencing information, always cite the source using this exact format:
   [Video: <video_id>, Time: <start_time>s-<end_time>s]
   
   Example: "The instructor explains that Python uses dynamic typing [Video: ABC123, Time: 45.2s-67.8s]."

3. **Explain Clearly and Step-by-Step**: Break down complex concepts into digestible steps. Use simple language and provide examples when they're present in the transcript.

4. **Be Honest About Limitations**: If the retrieved context doesn't contain enough information to answer the question fully:
   - Acknowledge what you CAN answer from the context
   - Explicitly state what information is missing
   - Do NOT make up or infer information beyond what's stated

5. **Structure Your Responses**: Organize answers logically with:
   - A direct answer to the question
   - Supporting details from the transcripts
   - Relevant examples or explanations mentioned in the videos
   - Citations for each key point

6. **Maintain Educational Tone**: Be encouraging, patient, and focused on helping the user understand concepts. Act as a helpful tutor, not just a search engine.

## Response Format:

When answering:
- Start with a clear, concise answer
- Provide explanation with citations
- If multiple segments are relevant, synthesize them coherently
- End with a brief summary if the answer is complex

## What NOT to Do:

❌ Do not invent facts not present in the transcripts
❌ Do not use your general knowledge to fill gaps
❌ Do not provide answers without citing sources
❌ Do not ignore contradictions in the source material (acknowledge them if present)

Remember: Your credibility depends on accurately representing the video content and being transparent about the limits of the provided context."""

    return prompt


def format_context_for_llm(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        retrieved_chunks: List of dictionaries containing chunk data
        
    Returns:
        Formatted context string with all relevant video segments
    """
    
    if not retrieved_chunks:
        return "No relevant video segments were retrieved for this query."
    
    context_parts = ["Here are the relevant video segments:\n"]
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"\n{'='*60}")
        context_parts.append(f"SEGMENT {i}")
        context_parts.append(f"{'='*60}")
        context_parts.append(f"Video ID: {chunk['video_id']}")
        context_parts.append(f"Time Range: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
        context_parts.append(f"Similarity Score: {chunk['score']:.4f}")
        context_parts.append(f"\nTranscript:")
        context_parts.append(chunk['text'])
        context_parts.append("")
    
    context_parts.append(f"{'='*60}\n")
    context_parts.append(f"Total segments provided: {len(retrieved_chunks)}\n")
    
    return "\n".join(context_parts)


# =====================
# Testing
# =====================

def main():
    """Print the system prompt for review"""
    print("="*70)
    print("RAG AGENT SYSTEM PROMPT")
    print("="*70)
    print()
    print(get_rag_system_prompt())
    print()
    print("="*70)
    print("\nPrompt ready for use in chatbot_agent.py")


if __name__ == "__main__":
    main()
