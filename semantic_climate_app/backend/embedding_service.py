"""
Local embedding generation via Ollama or sentence-transformers.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Updated 2025-12-10: Switched to all-mpnet-base-v2 via sentence-transformers.

Model selection based on sensitivity diagnostic comparing MiniLM, MPNet,
mxbai-embed-large, and nomic-embed-text for semantic climate metrics.

MPNet selected for:
- Highest velocity variance (most sensitive to local semantic shifts)
- Best local-vs-distant gap (preserves structural relationships)
- Valid α (DFA) values (not boundary condition like Nomic)
- Mid-range similarity (~0.22) — doesn't collapse distinctions

Retrieval-optimized models (Nomic, mxbai) collapse local semantic motion,
which undermines Δκ (curvature) and α (DFA) measurements.

See: scripts/embedding_sensitivity_diagnostic.py
"""

import numpy as np
from typing import List, Optional
import requests


class EmbeddingService:
    """
    Generate embeddings locally.

    Supports two backends:
    - sentence-transformers: Uses HuggingFace models directly (default, recommended)
    - ollama: Uses Ollama's embedding API
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        backend: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Model identifier
                - For sentence-transformers: "all-mpnet-base-v2" (default, recommended),
                  "all-MiniLM-L6-v2"
                - For ollama: "nomic-embed-text:v1.5", "mxbai-embed-large"
            backend: "sentence-transformers" (default) or "ollama"
            ollama_base_url: Ollama API base URL (only used with ollama backend)
        """
        self.model_name = model_name
        self.backend = backend
        self.ollama_base_url = ollama_base_url
        self._dimensions = None

        if backend == "ollama":
            self._init_ollama()
        elif backend == "sentence-transformers":
            self._init_sentence_transformers()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'sentence-transformers'")

    def _init_ollama(self):
        """Initialize Ollama backend."""
        print(f"Initializing Ollama embeddings: {self.model_name}...")

        # Verify model is available
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                # Check for exact match or partial match
                found = False
                for name in model_names:
                    if self.model_name in name or name in self.model_name:
                        found = True
                        break

                if not found:
                    print(f"  ⚠ Model '{self.model_name}' not found in Ollama.")
                    print(f"    Available models: {model_names}")
                    print(f"    Run: ollama pull {self.model_name}")
                else:
                    print(f"  ✓ Model available")
            else:
                print(f"  ⚠ Could not verify model availability")
        except requests.exceptions.ConnectionError:
            print(f"  ⚠ Ollama not running at {self.ollama_base_url}")
            print(f"    Start with: ollama serve")

        # Test embedding to get dimensions
        try:
            test_embedding = self._embed_ollama("test")
            self._dimensions = len(test_embedding)
            print(f"  ✓ Embedding dimensions: {self._dimensions}")
        except Exception as e:
            print(f"  ⚠ Could not verify embedding dimensions: {e}")
            self._dimensions = 768  # Default for nomic-embed-text

        print(f"✓ Ollama embedding service ready")

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers backend."""
        from sentence_transformers import SentenceTransformer

        print(f"Loading sentence-transformers model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self._dimensions = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded ({self._dimensions} dimensions)")

    def _embed_ollama(self, text: str) -> np.ndarray:
        """Generate embedding via Ollama API."""
        response = requests.post(
            f"{self.ollama_base_url}/api/embed",
            json={
                "model": self.model_name,
                "input": text
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama embedding failed: {response.text}")

        data = response.json()
        # Ollama returns {"embeddings": [[...]]} for single input
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"No embeddings returned: {data}")

        return np.array(embeddings[0], dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        if self.backend == "ollama":
            return self._embed_ollama(text)
        else:
            return self.model.encode(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors, shape (len(texts), dimensions)
        """
        if self.backend == "ollama":
            # Ollama supports batch embedding
            response = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": texts
                },
                timeout=60
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama batch embedding failed: {response.text}")

            data = response.json()
            embeddings = data.get("embeddings", [])
            return np.array(embeddings, dtype=np.float32)
        else:
            return self.model.encode(texts)

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions or 768
