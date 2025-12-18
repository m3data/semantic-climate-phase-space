"""
Ollama client for local LLM inference.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.
"""

import requests
from typing import List, Dict


class OllamaClient:
    """Client for Ollama local LLM server."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.history = []

    def chat(self, message: str) -> str:
        """
        Send a message and get response.

        Args:
            message: User message

        Returns:
            AI response text
        """
        self.history.append({"role": "user", "content": message})

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": False
        }

        try:
            # Extended timeout for very long texts (e.g., full chapters)
            response = requests.post(url, json=payload, timeout=600)  # 10 minutes
            response.raise_for_status()

            data = response.json()
            ai_message = data["message"]["content"]

            self.history.append({"role": "assistant", "content": ai_message})

            return ai_message

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def reset_history(self):
        """Clear conversation history."""
        self.history = []

    @staticmethod
    def get_available_models(base_url: str = "http://localhost:11434") -> List[Dict]:
        """
        Query Ollama server for available models.

        Returns:
            List of model info: [{"name": "llama2", "size": "3.8GB"}, ...]
        """
        try:
            url = f"{base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            return [
                {
                    "name": model["name"],
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", "")
                }
                for model in models
            ]
        except:
            return []

    @staticmethod
    def is_running(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"
