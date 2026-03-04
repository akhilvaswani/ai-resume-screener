"""
Configuration settings for the AI Resume Screener.
"""

import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # OpenAI
    "EMBEDDING_MODEL": "text-embedding-3-small",

    # Caching
    "CACHE_DB": "embeddings_cache.db",
    "MAX_INPUT_CHARS": 30000,

    # Scoring weights
    "SEMANTIC_WEIGHT": 0.6,
    "SKILL_WEIGHT": 0.4,

    # Thresholds
    "SIMILARITY_THRESHOLD": 0.7,
    "POTENTIAL_THRESHOLD": 0.5,

    # Streamlit
    "PAGE_TITLE": "AI Resume Screener",
}
