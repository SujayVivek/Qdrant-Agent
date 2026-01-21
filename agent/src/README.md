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