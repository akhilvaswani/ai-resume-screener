"""
AI Resume Screener - Embedding Manager
Handles OpenAI embedding generation with local SQLite caching
and rate limiting.
"""

import os
import json
import time
import sqlite3
import hashlib
from openai import OpenAI

from config import CONFIG


class EmbeddingManager:
    """
    Manages text embeddings with caching and rate limiting.
    Uses OpenAI's embedding API and caches results in SQLite
    to avoid redundant API calls.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = CONFIG["EMBEDDING_MODEL"]
        self.cache_db = CONFIG["CACHE_DB"]
        self.max_retries = 3
        self.retry_delay = 1.0

        self._init_cache()

    def _init_cache(self):
        """Initialize the SQLite cache database."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT,
                embedding TEXT,
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _get_text_hash(self, text):
        """Generate a hash for caching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _check_cache(self, text_hash):
        """Look up an embedding in the cache."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?",
            (text_hash, self.model)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def _save_cache(self, text_hash, embedding, token_count):
        """Save an embedding to the cache."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, token_count)
               VALUES (?, ?, ?, ?)""",
            (text_hash, self.model, json.dumps(embedding), token_count)
        )
        conn.commit()
        conn.close()

    def get_embedding(self, text):
        """
        Get the embedding for a piece of text.
        Returns cached result if available, otherwise calls the API.

        Args:
            text: The text to embed

        Returns:
            List of floats (the embedding vector), or None on error
        """
        if not text or not text.strip():
            return None

        # Truncate to model's max token limit (roughly 4 chars per token)
        max_chars = CONFIG.get("MAX_INPUT_CHARS", 30000)
        if len(text) > max_chars:
            text = text[:max_chars]

        text_hash = self._get_text_hash(text)

        # Check cache first
        cached = self._check_cache(text_hash)
        if cached is not None:
            return cached

        # Call the API with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                token_count = response.usage.total_tokens

                # Cache the result
                self._save_cache(text_hash, embedding, token_count)

                return embedding

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"  Embedding API error (attempt {attempt + 1}): {e}")
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Embedding API failed after {self.max_retries} attempts: {e}")
                    return None

    def get_embeddings_batch(self, texts):
        """
        Get embeddings for multiple texts efficiently.
        Uses caching and batches uncached texts into a single API call.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (same order as input)
        """
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            text_hash = self._get_text_hash(text)
            cached = self._check_cache(text_hash)

            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results

        # Batch API call for uncached texts
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=uncached_texts
            )

            for j, embedding_data in enumerate(response.data):
                idx = uncached_indices[j]
                embedding = embedding_data.embedding
                results[idx] = embedding

                # Cache each result
                text_hash = self._get_text_hash(uncached_texts[j])
                self._save_cache(text_hash, embedding, 0)

        except Exception as e:
            print(f"Batch embedding error: {e}")
            # Fall back to individual calls
            for j, text in enumerate(uncached_texts):
                idx = uncached_indices[j]
                results[idx] = self.get_embedding(text)

        return results

    def get_cache_stats(self):
        """Get statistics about the embedding cache."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(token_count) FROM embeddings")
        total_tokens = cursor.fetchone()[0] or 0

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM embeddings")
        row = cursor.fetchone()

        conn.close()

        return {
            "total_cached": total,
            "total_tokens": total_tokens,
            "oldest_entry": row[0],
            "newest_entry": row[1]
        }

    def clear_cache(self):
        """Clear the entire embedding cache."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        print("Embedding cache cleared")
