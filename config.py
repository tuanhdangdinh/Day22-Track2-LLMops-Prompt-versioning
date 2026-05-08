"""Shared configuration helpers — loads .env and validates required keys."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# LangSmith
LANGSMITH_API_KEY  = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT  = os.environ.get("LANGCHAIN_PROJECT", "day22-rag-lab")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# OpenAI / compatible endpoint
OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL      = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL         = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def setup_langsmith():
    """Set LangSmith environment variables before any LangChain imports."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]   = LANGCHAIN_ENDPOINT


def check_config():
    missing = []
    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")


if __name__ == "__main__":
    try:
        check_config()
        print("✅ Config loaded successfully")
        print(f"   LangSmith project : {LANGCHAIN_PROJECT}")
        print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
        print(f"   Default LLM model : {OPENAI_MODEL}")
        print(f"   Embedding model   : {OPENAI_EMBEDDING_MODEL}")
    except EnvironmentError as e:
        print(f"❌ {e}")
        print("   Copy .env.example to .env and fill in your keys.")
