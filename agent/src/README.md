## Sanity Check
To check if all APIs are set/not
Run: python sanity_check.py

## Transcript Loader
Load video id: https://www.youtube.com/watch?v=Q2iosMpx0Qw
Video ID = Q2iosMpx0Qw
Gives transcript at ../data/transcripts/raw/
Run: pytohon transcript_loader.py

## Chunker
Chunk the raw data in a custom duration
Gives processed data at: ../data/transcripts/processed
RUN: python chunk_transcripts.py ../data/transcripts/raw/Q2iosMpx0Qw.json 45

## Qdrant Setup
Creates a collection, with some configuration eg: vector size 384
Run after a container of qdrant is running 

1. docker pull qdrant/qdrant
2. docker run -p 6333:6333 qdrant/qdrant

Look for it on localhost:6333

### Embedding Model: Use OpenAI - Pay $$ ffs
Run a single file: python src/embed_chunks.py data/transcripts/processed/Q2iosMpx0Qw_chunked.json

Run all chunked files: python src/embed_chunks.py all

## agent_prompt.py
This will comprise of all the prompts, you can run
python agent_prompt.py to review the PROMPT!

## retriever.py
Testing can be done by running it : 
python retriever.py "Name? What are you buildling? What is your age"

- It searches for keywords Name, building and age in the qdrant collection

## chatbot_agent.py
A reasoning model for Claude to respond to!
Completes the RAG Application here Yay! Working too

## run_chatbot.py
Orchestrates everything, and runs like a charm! It provides a continuous question loop, shows retrieved video segments with video_id etc!

## Concept Memory.py
This goes ahead and creates a collection for concept memory
Need to run once only

## Concept Utils
Has utility functions to add points, query points from concept memory


