#!/usr/bin/env python3
"""
YouTube Transcript Loader
Fetches transcripts from YouTube videos and saves them as JSON with timestamps
"""

import json
import sys
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript for a YouTube video
    
    Args:
        video_id: YouTube video ID (e.g., 'dQw4w9WgXcQ')
    
    Returns:
        List of transcript entries with start time, duration, and text
    
    Raises:
        Exception if transcript is unavailable
    """
    try:
        api = YouTubeTranscriptApi()

        fetched = api.fetch(video_id)  
        transcript = [
            {
                "text": s.text,
                "start": s.start,
                "duration": s.duration
            }
            for s in fetched.snippets
        ]

        return transcript
    except TranscriptsDisabled:
        raise Exception(f"Transcripts are disabled for video: {video_id}")
    except NoTranscriptFound:
        raise Exception(f"No transcript found for video: {video_id}")
    except VideoUnavailable:
        raise Exception(f"Video unavailable: {video_id}")
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")


def save_transcript(video_id: str, transcript: list[dict], output_dir: Path = None) -> Path:
    """
    Save transcript to JSON file
    
    Args:
        video_id: YouTube video ID
        transcript: List of transcript entries
        output_dir: Output directory (defaults to data/transcripts/raw/)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        # Get project root (go up from src/ to agent/)
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "transcripts" / "raw"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transcript
    output_file = output_dir / f"{video_id}.json"
    
    data = {
        "video_id": video_id,
        "total_entries": len(transcript),
        "transcript": transcript
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_file


def load_transcript(video_id: str) -> dict:
    """
    Load transcript and save it to disk
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary with video_id, total_entries, and transcript
    """
    print(f"📥 Fetching transcript for video: {video_id}")
    
    try:
        # Fetch transcript
        transcript = fetch_transcript(video_id)
        print(f"   ✓ Fetched {len(transcript)} transcript entries")
        
        # Save to disk
        output_file = save_transcript(video_id, transcript)
        print(f"   ✓ Saved to: {output_file}")
        
        return {
            "video_id": video_id,
            "total_entries": len(transcript),
            "transcript": transcript
        }
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise


def main():
    """
    Main entry point for CLI usage
    """
    if len(sys.argv) < 2:
        print("Usage: python transcript_loader.py <video_id>")
        print("Example: python transcript_loader.py dQw4w9WgXcQ")
        sys.exit(1)
    
    video_id = sys.argv[1]
    
    try:
        result = load_transcript(video_id)
        print(f"\n✅ Successfully processed video: {video_id}")
        print(f"   Total entries: {result['total_entries']}")
    except Exception as e:
        print(f"\n❌ Failed to process video: {video_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
