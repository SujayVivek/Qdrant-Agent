#!/usr/bin/env python3
"""
Transcript Chunker
Converts raw transcripts into time-aware chunks for embedding and vector storage
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import hashlib


def load_raw_transcript(file_path: Path) -> Dict:
    """
    Load raw transcript JSON file
    
    Args:
        file_path: Path to raw transcript JSON
    
    Returns:
        Dictionary with video_id and transcript entries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_chunk_id(video_id: str, start_time: float) -> str:
    """
    Generate deterministic chunk ID
    
    Args:
        video_id: YouTube video ID
        start_time: Start time of chunk
    
    Returns:
        Unique, deterministic chunk ID
    """
    # Create deterministic ID based on video_id and start_time
    content = f"{video_id}_{start_time:.2f}"
    hash_object = hashlib.md5(content.encode())
    return f"{video_id}_{hash_object.hexdigest()[:12]}"


def chunk_transcript(transcript_entries: List[Dict], video_id: str, target_duration: float = 45.0) -> List[Dict]:
    """
    Group transcript entries into time-based chunks
    
    Args:
        transcript_entries: List of transcript entries with start, duration, text
        video_id: YouTube video ID
        target_duration: Target duration for each chunk in seconds (default: 45s)
    
    Returns:
        List of chunked transcript objects
    """
    chunks = []
    current_chunk = []
    chunk_start_time = None
    chunk_duration = 0.0
    
    for entry in transcript_entries:
        start = entry['start']
        duration = entry['duration']
        text = entry['text']
        
        # Initialize first chunk
        if chunk_start_time is None:
            chunk_start_time = start
        
        # Add entry to current chunk
        current_chunk.append(text)
        chunk_duration = (start + duration) - chunk_start_time
        
        # Check if we've reached target duration
        if chunk_duration >= target_duration:
            # Finalize current chunk
            chunk_end_time = start + duration
            chunk_text = ' '.join(current_chunk).strip()
            
            chunk_obj = {
                'chunk_id': generate_chunk_id(video_id, chunk_start_time),
                'video_id': video_id,
                'start_time': round(chunk_start_time, 2),
                'end_time': round(chunk_end_time, 2),
                'duration': round(chunk_duration, 2),
                'text': chunk_text
            }
            chunks.append(chunk_obj)
            
            # Reset for next chunk
            current_chunk = []
            chunk_start_time = None
            chunk_duration = 0.0
    
    # Handle remaining entries
    if current_chunk:
        last_entry = transcript_entries[-1]
        chunk_end_time = last_entry['start'] + last_entry['duration']
        chunk_text = ' '.join(current_chunk).strip()
        
        chunk_obj = {
            'chunk_id': generate_chunk_id(video_id, chunk_start_time),
            'video_id': video_id,
            'start_time': round(chunk_start_time, 2),
            'end_time': round(chunk_end_time, 2),
            'duration': round(chunk_duration, 2),
            'text': chunk_text
        }
        chunks.append(chunk_obj)
    
    return chunks


def save_chunked_transcript(video_id: str, chunks: List[Dict], output_dir: Path = None) -> Path:
    """
    Save chunked transcript to JSON file
    
    Args:
        video_id: YouTube video ID
        chunks: List of chunk objects
        output_dir: Output directory (defaults to data/transcripts/processed/)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        # Get project root (go up from src/ to agent/)
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "transcripts" / "processed"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunked transcript
    output_file = output_dir / f"{video_id}_chunked.json"
    
    data = {
        'video_id': video_id,
        'total_chunks': len(chunks),
        'chunks': chunks
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_file


def process_transcript(input_file: Path, target_duration: float = 45.0) -> Dict:
    """
    Process a raw transcript file into chunks
    
    Args:
        input_file: Path to raw transcript JSON
        target_duration: Target chunk duration in seconds
    
    Returns:
        Dictionary with video_id, total_chunks, and chunks
    """
    print(f"📝 Processing transcript: {input_file.name}")
    
    # Load raw transcript
    raw_data = load_raw_transcript(input_file)
    video_id = raw_data['video_id']
    transcript_entries = raw_data['transcript']
    
    print(f"   → Video ID: {video_id}")
    print(f"   → Raw entries: {len(transcript_entries)}")
    
    # Chunk transcript
    chunks = chunk_transcript(transcript_entries, video_id, target_duration)
    print(f"   → Created {len(chunks)} chunks (~{target_duration}s each)")
    
    # Save chunked transcript
    output_file = save_chunked_transcript(video_id, chunks)
    print(f"   ✓ Saved to: {output_file}")
    
    return {
        'video_id': video_id,
        'total_chunks': len(chunks),
        'chunks': chunks
    }


def main():
    """
    Main entry point for CLI usage
    """
    if len(sys.argv) < 2:
        print("Usage: python chunk_transcripts.py <raw_transcript_file.json> [target_duration]")
        print("Example: python chunk_transcripts.py data/transcripts/raw/video123.json 45")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    target_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    
    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    try:
        result = process_transcript(input_file, target_duration)
        print(f"\n✅ Successfully processed: {result['video_id']}")
        print(f"   Total chunks: {result['total_chunks']}")
    except Exception as e:
        print(f"\n❌ Failed to process transcript: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
