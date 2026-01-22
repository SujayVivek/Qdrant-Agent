# Video AI Tutor - Streamlit Frontend

ChatGPT-like interface for learning from video transcripts with persistent memory.

## 🚀 Setup

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Environment Variables

Make sure you have a `.env` file in the `agent` directory with:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
QDRANT_URL=http://localhost:6333
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## ✨ Features

### Video Processing
- Add multiple YouTube video URLs
- Process videos with one click
- Automatic transcript fetching, chunking, and embedding
- Session-based video tracking

### Chat Interface
- ChatGPT-like conversation experience
- Context-aware answers with timestamp citations
- Display of retrieved video segments
- Clean and intuitive UI

### Learning Memory
- 👍/👎 feedback collection
- Automatic concept learning from positive feedback
- Improved retrieval over time
- Persistent memory across sessions

### Session Management
- Unique session IDs for each chat
- "New Chat" button to start fresh
- Video status tracking
- Robust error handling

## 📊 Architecture

```
frontend/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file

Imports from:
../agent/src/          # Backend logic
├── qdrant_setup.py    # Collection setup
├── transcript_loader.py
├── chunk_transcripts.py
├── chatbot_agent.py
├── feedback_handler.py
└── config.py
```

## 🎯 Usage Flow

1. **Add Videos**: Paste YouTube URLs in sidebar
2. **Process Videos**: Click "Process Videos" to ingest
3. **Ask Questions**: Chat with the AI tutor
4. **Provide Feedback**: Help the system learn with 👍/👎
5. **New Chat**: Start fresh anytime

## 🔧 Configuration

- **Collection**: `video_segments_test` (for video chunks)
- **Memory**: `concept_memory` (for learning)
- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dims)
- **LLM**: Claude Sonnet 3.5 via Anthropic

## 🐛 Troubleshooting

### Qdrant Connection Error
- Ensure Qdrant is running: `docker ps`
- Check `QDRANT_URL` in `.env`

### API Key Errors
- Verify `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `.env`
- Check API key validity

### Video Processing Fails
- Verify YouTube URL is valid
- Check if video has captions/transcripts available
- Some videos may have restricted transcripts

## 📝 Notes

- Videos are tagged with session IDs
- Each session maintains its own conversation history
- Concept memory is shared across all sessions (improves over time)
- The app is designed for hackathon demos - optimized for clarity and simplicity
