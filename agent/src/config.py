import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve required environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Validate that all required variables are present
missing_vars = []
if not ANTHROPIC_API_KEY:
    missing_vars.append("ANTHROPIC_API_KEY")
if not HF_API_KEY:
    missing_vars.append("HF_API_KEY")
if not QDRANT_URL:
    missing_vars.append("QDRANT_URL")

if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}. Please set them in your .env file.")
