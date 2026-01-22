#!/usr/bin/env python3
"""
Streamlit Frontend for Video-Based AI Tutor
ChatGPT-like interface for learning from video transcripts.
"""

import sys
import os
from pathlib import Path
import uuid
import streamlit as st
from datetime import datetime
import uuid

# Add agent/src directory to path
agent_src_dir = Path(__file__).resolve().parent.parent / "agent" / "src"
sys.path.insert(0, str(agent_src_dir))

# Import agent modules
from qdrant_setup import create_collection_if_not_exists, COLLECTIONS
from transcript_loader import fetch_transcript, save_transcript
from chunk_transcripts import chunk_transcript
from chatbot_agent import VideoChatbotAgent
from feedback_handler import handle_feedback
from config import QDRANT_URL

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct
from langchain_openai import OpenAIEmbeddings


# =====================
# Configuration
# =====================

st.set_page_config(
    page_title="Video AI Tutor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================
# Session State Initialization
# =====================

def init_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "video_urls" not in st.session_state:
        st.session_state.video_urls = []
    
    if "processed_videos" not in st.session_state:
        st.session_state.processed_videos = []
    
    if "videos_ready" not in st.session_state:
        st.session_state.videos_ready = False
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "qdrant_setup_done" not in st.session_state:
        st.session_state.qdrant_setup_done = False
    
    if "last_retrieved_chunks" not in st.session_state:
        st.session_state.last_retrieved_chunks = []
    
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


# =====================
# Qdrant Setup
# =====================

def ensure_qdrant_collections():
    """Ensure all required Qdrant collections exist"""
    if st.session_state.qdrant_setup_done:
        return True
    
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        for collection_name, config in COLLECTIONS.items():
            create_collection_if_not_exists(
                client=client,
                collection_name=collection_name,
                vector_size=config["vector_size"],
                distance=config["distance"]
            )
        
        st.session_state.qdrant_setup_done = True
        return True
    except Exception as e:
        st.error(f"Failed to setup Qdrant: {e}")
        return False


# =====================
# Video Processing
# =====================

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    else:
        return url


def process_video(video_url: str, progress_container) -> bool:
    """Process a single video: fetch transcript, chunk, embed, and store"""
    try:
        video_id = extract_video_id(video_url)
        print(f"\n{'='*70}")
        print(f"Processing video: {video_id}")
        print(f"{'='*70}")
        
        # Step 1: Fetch transcript
        progress_container.info(f"📥 Fetching transcript for video: {video_id}")
        print(f"\n[1/3] Fetching transcript...")
        
        progress_bar = progress_container.progress(0, text="Fetching transcript...")
        
        # fetch_transcript returns a list of dict with 'text', 'start', 'duration'
        transcript_entries = fetch_transcript(video_id)
        
        if not transcript_entries:
            progress_container.error(f"❌ Failed to fetch transcript for {video_id}")
            print(f"ERROR: Failed to fetch transcript for {video_id}")
            return False
        
        progress_bar.progress(33, text=f"✓ Fetched {len(transcript_entries)} entries")
        progress_container.success(f"✓ Fetched {len(transcript_entries)} transcript entries")
        print(f"✓ Fetched {len(transcript_entries)} transcript entries")
        
        # Step 2: Chunk transcript
        progress_bar.progress(33, text="Chunking transcript...")
        progress_container.info(f"✂️ Chunking transcript...")
        print(f"\n[2/3] Chunking transcript...")
        
        # chunk_transcript takes transcript_entries, video_id, and target_duration
        chunks = chunk_transcript(
            transcript_entries=transcript_entries,
            video_id=video_id,
            target_duration=60.0
        )
        
        progress_bar.progress(66, text=f"✓ Created {len(chunks)} chunks")
        progress_container.success(f"✓ Created {len(chunks)} chunks")
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Embed and store
        progress_bar.progress(66, text="Embedding and storing chunks...")
        progress_container.info(f"🔄 Embedding {len(chunks)} chunks...")
        print(f"\n[3/3] Embedding and storing {len(chunks)} chunks...")
        
        # Initialize clients
        qdrant_client = QdrantClient(url=QDRANT_URL)
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Process in batches
        batch_size = 50
        total_inserted = 0
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"Processing in {num_batches} batches of {batch_size} chunks each...")
        
        for batch_idx, i in enumerate(range(0, len(chunks), batch_size), 1):
            batch = chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            print(f"  Batch {batch_idx}/{num_batches}: Embedding {len(batch)} chunks...")
            
            # Generate embeddings
            embeddings = embeddings_model.embed_documents(texts)
            
            # Create points
            points = []
            for chunk, embedding in zip(batch, embeddings):
                points.append(
                    PointStruct(
                        id = str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "video_id": chunk["video_id"],
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                            "text": chunk["text"],
                            "session_id": st.session_state.session_id  # Tag with session
                        }
                    )
                )
            
            # Insert into Qdrant
            qdrant_client.upsert(
                collection_name="video_segments_test",
                points=points
            )
            
            total_inserted += len(points)
            
            # Update progress
            embed_progress = 66 + int((batch_idx / num_batches) * 34)
            progress_bar.progress(embed_progress, text=f"Embedded {total_inserted}/{len(chunks)} chunks")
            progress_container.info(f"   Embedded and inserted {total_inserted}/{len(chunks)} chunks...")
            print(f"  Batch {batch_idx}/{num_batches}: ✓ Inserted {len(points)} chunks (Total: {total_inserted}/{len(chunks)})")
        
        progress_bar.progress(100, text="✅ Complete!")
        progress_container.success(f"✅ Processed {video_id}: {total_inserted} chunks stored")
        print(f"\n✅ Successfully processed {video_id}")
        print(f"   Total chunks embedded and stored: {total_inserted}")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        progress_container.error(f"❌ Error processing video: {e}")
        print(f"\n❌ ERROR processing video: {e}")
        import traceback
        error_trace = traceback.format_exc()
        progress_container.error(error_trace)
        print(error_trace)
        return False


# =====================
# UI Components
# =====================

def render_sidebar():
    """Render sidebar with video management"""
    with st.sidebar:
        st.title("🎥 Video AI Tutor")
        st.markdown(f"**Session:** `{st.session_state.session_id}`")
        st.divider()
        
        # New Chat button
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.video_urls = []
            st.session_state.processed_videos = []
            st.session_state.videos_ready = False
            st.session_state.last_retrieved_chunks = []
            st.session_state.last_question = ""
            st.rerun()
        
        st.divider()
        
        # Video URL input
        st.subheader("📹 Add Videos")
        video_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
            key="video_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add", use_container_width=True):
                if video_url and video_url not in st.session_state.video_urls:
                    st.session_state.video_urls.append(video_url)
                    st.success("Added!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.video_urls = []
                st.session_state.processed_videos = []
                st.session_state.videos_ready = False
                st.rerun()
        
        # Display added videos
        if st.session_state.video_urls:
            st.markdown("**Added Videos:**")
            for i, url in enumerate(st.session_state.video_urls, 1):
                video_id = extract_video_id(url)
                status = "✅" if video_id in st.session_state.processed_videos else "⏳"
                st.text(f"{status} {i}. {video_id}")
        
        st.divider()
        
        # Process Videos button
        if st.session_state.video_urls:
            if st.button("🚀 Process Videos", use_container_width=True, type="primary"):
                print("\n" + "="*70)
                print("STARTING VIDEO PROCESSING")
                print(f"Total videos to process: {len(st.session_state.video_urls)}")
                print("="*70)
                
                progress_container = st.container()
                
                overall_progress = st.progress(0, text="Starting video processing...")
                
                for idx, video_url in enumerate(st.session_state.video_urls, 1):
                    video_id = extract_video_id(video_url)
                    
                    print(f"\n[Video {idx}/{len(st.session_state.video_urls)}] Processing: {video_id}")
                    
                    if video_id not in st.session_state.processed_videos:
                        overall_progress.progress(
                            int((idx - 1) / len(st.session_state.video_urls) * 100),
                            text=f"Processing video {idx}/{len(st.session_state.video_urls)}: {video_id}"
                        )
                        
                        success = process_video(video_url, progress_container)
                        
                        if success:
                            st.session_state.processed_videos.append(video_id)
                            print(f"✓ Video {idx}/{len(st.session_state.video_urls)} completed successfully")
                        else:
                            print(f"✗ Video {idx}/{len(st.session_state.video_urls)} failed")
                    else:
                        print(f"⊙ Video {idx}/{len(st.session_state.video_urls)} already processed, skipping")
                
                overall_progress.progress(100, text="All videos processed!")
                
                # Check if all videos processed
                if len(st.session_state.processed_videos) == len(st.session_state.video_urls):
                    st.session_state.videos_ready = True
                    st.success("✅ All videos processed!")
                    print("\n" + "="*70)
                    print("✅ ALL VIDEOS PROCESSED SUCCESSFULLY")
                    print("="*70 + "\n")
                    st.rerun()
                else:
                    print("\n⚠️ Some videos failed to process")
                    st.warning(f"⚠️ {len(st.session_state.processed_videos)}/{len(st.session_state.video_urls)} videos processed")
        
        # Status indicator
        st.divider()
        st.subheader("📊 Status")
        if st.session_state.videos_ready:
            st.success("✅ Ready to chat!")
        else:
            st.warning("⏳ Add and process videos first")


def render_chat():
    """Render chat interface"""
    st.title("💬 Chat with Your Videos")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show retrieved chunks for assistant messages
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("📚 Retrieved Video Segments"):
                    for i, chunk in enumerate(message["chunks"], 1):
                        st.markdown(f"""
                        **[{i}]** Video: `{chunk['video_id']}` | 
                        Time: `{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s` | 
                        Score: `{chunk['score']:.3f}`
                        
                        > {chunk['text'][:150]}...
                        """)
    
    # Chat input
    if st.session_state.videos_ready:
        if prompt := st.chat_input("Ask a question about your videos..."):
            print("\n" + "="*70)
            print(f"USER QUESTION: {prompt}")
            print("="*70)
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.last_question = prompt
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get answer from agent
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                
                status_placeholder.info("🔍 Searching for relevant video segments...")
                print("\n[1/3] Retrieving relevant chunks from Qdrant...")
                
                # Initialize agent if needed
                if st.session_state.agent is None:
                    print("Initializing chatbot agent...")
                    st.session_state.agent = VideoChatbotAgent(top_k=5)
                    print("✓ Agent initialized")
                
                status_placeholder.info("🧠 Generating answer with Claude...")
                print("[2/3] Generating answer with Claude...")
                
                # Get answer
                answer, chunks = st.session_state.agent.answer_question(prompt)
                
                print(f"✓ Retrieved {len(chunks)} relevant chunks")
                print("[3/3] Formatting response...")
                
                # Store chunks for feedback
                st.session_state.last_retrieved_chunks = chunks
                
                status_placeholder.empty()
                
                # Display answer
                st.markdown(answer)
                print(f"\n✓ Answer generated and displayed")
                print("="*70 + "\n")
                
                # Show retrieved chunks
                with st.expander("📚 Retrieved Video Segments"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"""
                        **[{i}]** Video: `{chunk['video_id']}` | 
                        Time: `{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s` | 
                        Score: `{chunk['score']:.3f}`
                        
                        > {chunk['text'][:150]}...
                        """)
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks
                })
            
            st.rerun()
    else:
        st.info("👆 Please add and process videos in the sidebar first.")


def render_feedback():
    """Render feedback buttons"""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.divider()
        st.subheader("💭 Was this helpful?")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("👍 Yes", use_container_width=True):
                print("\n[Feedback] User provided positive feedback")
                print("\n[Feedback] User provided positive feedback")
                # Process positive feedback
                try:
                    print("Processing positive feedback...")
                    result = handle_feedback(
                        user_question=st.session_state.last_question,
                        retrieved_chunks=st.session_state.last_retrieved_chunks,
                        user_feedback=True,
                        qdrant_client=st.session_state.agent.qdrant_client,
                        embeddings_model=st.session_state.agent.embeddings_model
                    )
                    
                    if result['action'] == 'created':
                        st.success(f"✅ Learned new concept!")
                        print(f"✓ Created new concept in memory")
                    elif result['action'] == 'updated':
                        st.success(f"✅ Reinforced concept!")
                        print(f"✓ Updated existing concept in memory")
                    print(f"Result: {result}")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    print(f"✗ Error processing feedback: {e}")
        
        with col2:
            if st.button("👎 No", use_container_width=True):
                st.info("📝 Feedback recorded. Memory not updated.")
                print("\n[Feedback] User provided negative feedback (memory not updated)")


# =====================
# Main App
# =====================

def main():
    """Main application entry point"""
    
    print("\n" + "="*70)
    print("VIDEO AI TUTOR - STREAMLIT FRONTEND")
    print("="*70)
    
    # Initialize session state
    init_session_state()
    print(f"Session ID: {st.session_state.session_id}")
    
    # Ensure Qdrant collections exist
    if not st.session_state.qdrant_setup_done:
        print("\nChecking Qdrant collections...")
        with st.spinner("Setting up Qdrant collections..."):
            ensure_qdrant_collections()
        print("✓ Qdrant collections ready")
    
    # Render UI
    render_sidebar()
    render_chat()
    
    # Render feedback only if there are messages
    if st.session_state.messages:
        render_feedback()


if __name__ == "__main__":
    main()
