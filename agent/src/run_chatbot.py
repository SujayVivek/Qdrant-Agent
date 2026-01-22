#!/usr/bin/env python3
"""
Interactive CLI Runner for Video Transcript Chatbot
Simple, demo-friendly interface for judges and users.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatbot_agent import VideoChatbotAgent


def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("🎥 VIDEO TRANSCRIPT CHATBOT")
    print("="*70)
    print("Ask questions about the video transcripts in the database.")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*70 + "\n")


def print_retrieved_segments(chunks):
    """Print retrieved video segments in a clean format"""
    print("\n" + "─"*70)
    print("📚 RETRIEVED VIDEO SEGMENTS")
    print("─"*70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] Video ID: {chunk['video_id']}")
        print(f"    Time: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
        print(f"    Relevance Score: {chunk['score']:.4f}")
        
        # Show a preview of the text (first 150 characters)
        text_preview = chunk['text'][:150]
        if len(chunk['text']) > 150:
            text_preview += "..."
        print(f"    Preview: {text_preview}")
    
    print("\n" + "─"*70)


def print_answer(answer):
    """Print the agent's answer"""
    print("\n" + "="*70)
    print("💡 ANSWER")
    print("="*70)
    print(f"\n{answer}\n")
    print("="*70)


def main():
    """Main interactive loop"""
    
    print_header()
    
    try:
        # Initialize the chatbot agent
        print("🚀 Initializing chatbot agent...\n")
        agent = VideoChatbotAgent(top_k=5)
        print("✅ Agent ready!\n")
        
        # Interactive loop
        while True:
            try:
                # Get user question
                print("─"*70)
                user_question = input("\n🙋 Your Question: ").strip()
                
                # Check for exit commands
                if user_question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thanks for using the Video Transcript Chatbot!")
                    print("="*70 + "\n")
                    break
                
                # Skip empty questions
                if not user_question:
                    print("⚠️  Please enter a question.")
                    continue
                
                # Get answer from agent
                print(f"\n🔍 Processing your question...\n")
                answer, chunks = agent.answer_question(user_question)
                
                # Display results
                print_retrieved_segments(chunks)
                print_answer(answer)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Video Transcript Chatbot!")
                print("="*70 + "\n")
                break
                
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")
                print("Please try again with a different question.\n")
                continue
    
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
