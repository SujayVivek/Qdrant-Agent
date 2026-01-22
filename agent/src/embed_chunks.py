#!/usr/bin/env python3
"""
Embed Transcript Chunks (OpenAI via LangChain)
Generates embeddings for transcript chunks using OpenAI's text-embedding-3-small model
via LangChain and stores them in Qdrant.
"""
import uuid
import json
import sys
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QDRANT_URL


# =====================
# Configuration
# =====================

COLLECTION_NAME = "video_segments_test"
BATCH_SIZE = 50  # OpenAI can handle larger batches efficiently
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536


# =====================
# Helpers
# =====================

def load_chunked_transcript(file_path: Path) -> Dict:
    """Load chunked transcript JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: List[str], embeddings_model: OpenAIEmbeddings) -> List[List[float]]:
    """
    Generate embeddings using OpenAI via LangChain
    Handles API errors gracefully
    """
    try:
        embeddings = embeddings_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        print(f"\n⚠️  Error generating embeddings: {e}")
        raise


def create_points(chunks: List[Dict], embeddings: List[List[float]]) -> List[PointStruct]:
    """Create Qdrant PointStructs with metadata"""
    points = []

    for chunk, embedding in zip(chunks, embeddings):
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
                },
            )
        )

    return points


def process_file(file_path: Path, qdrant_client: QdrantClient, embeddings_model: OpenAIEmbeddings) -> int:
    """Process a single chunked transcript file"""
    print(f"\n📄 Processing: {file_path.name}")

    try:
        data = load_chunked_transcript(file_path)
        chunks = data["chunks"]
        video_id = data["video_id"]

        print(f"   → Video ID: {video_id}")
        print(f"   → Total chunks: {len(chunks)}")

        total_inserted = 0

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Embedding & inserting"):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]

            try:
                embeddings = embed_texts(texts, embeddings_model)
                points = create_points(batch, embeddings)

                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )

                total_inserted += len(points)

            except Exception as e:
                print(f"\n⚠️  Error processing batch {i // BATCH_SIZE + 1}: {e}")
                print(f"   Skipping batch and continuing...")
                continue

        print(f"   ✓ Inserted {total_inserted} vectors")
        return total_inserted

    except Exception as e:
        print(f"\n❌ Error processing file {file_path.name}: {e}")
        return 0


# =====================
# Main
# =====================

def main():
    if len(sys.argv) < 2:
        print("Usage: python embed_chunks.py <chunked_transcript.json>")
        print("       python embed_chunks.py all")
        sys.exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set it in your .env file")
        sys.exit(1)

    print(f"🚀 Starting embedding process (OpenAI {EMBEDDING_MODEL})...")
    print(f"   Vector dimension: {VECTOR_DIMENSION}")

    try:
        # Initialize OpenAI embeddings via LangChain
        print("\n🤖 Initializing OpenAI embeddings...")
        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        print("   ✓ OpenAI embeddings initialized")

        # Connect to Qdrant
        print(f"\n🔗 Connecting to Qdrant: {QDRANT_URL}")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        print("   ✓ Connected")

        # Determine files to process
        if sys.argv[1].lower() == "all":
            project_root = Path(__file__).resolve().parent.parent
            processed_dir = project_root / "data" / "transcripts" / "processed"
            files = list(processed_dir.glob("*_chunked.json"))

            if not files:
                print(f"\n⚠️  No chunked transcript files found in {processed_dir}")
                sys.exit(1)

            print(f"\n📚 Found {len(files)} file(s) to process")
        else:
            files = [Path(sys.argv[1])]
            if not files[0].exists():
                print(f"\n❌ File not found: {files[0]}")
                sys.exit(1)

        # Process all files
        total_chunks = 0
        successful_files = 0

        for file_path in files:
            chunks_processed = process_file(file_path, qdrant_client, embeddings_model)
            if chunks_processed > 0:
                total_chunks += chunks_processed
                successful_files += 1

        print("\n✅ Embedding complete!")
        print(f"   Files processed successfully: {successful_files}/{len(files)}")
        print(f"   Total chunks embedded: {total_chunks}")
        print(f"   Collection: {COLLECTION_NAME}")

    except Exception as e:
        print(f"\n❌ Failed to embed chunks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
