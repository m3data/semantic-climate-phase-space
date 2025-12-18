"""
LLM Client implementations for multiple providers.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Supports:
- Ollama (Local models) - existing, for privacy/offline
- Together AI - fast cloud inference (OpenAI-compatible API)
- Anthropic (Claude) - for future use
- OpenAI (GPT-4, etc.) - for future use
"""

import os
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMClient(ABC):
    """Base class for all LLM clients."""

    @abstractmethod
    def chat(self, message: str) -> str:
        """Send a message and get response."""
        pass

    @abstractmethod
    def reset_history(self):
        """Clear conversation history."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for display."""
        pass


class OllamaClient(LLMClient):
    """Client for Ollama local LLM server."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.history = []

    @property
    def provider_name(self) -> str:
        return "Ollama (Local)"

    def chat(self, message: str) -> str:
        """Send a message and get response."""
        self.history.append({"role": "user", "content": message})

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": False
        }

        try:
            # Extended timeout for local inference
            response = requests.post(url, json=payload, timeout=600)
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
        """Query Ollama server for available models."""
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


class TogetherClient(LLMClient):
    """
    Client for Together AI cloud inference.

    Together AI provides fast cloud inference with OpenAI-compatible API.
    Typical latency: 2-5 seconds vs 30-120s for local Ollama.

    Models available include:
    - meta-llama/Llama-3.2-3B-Instruct (fast, good for experiments)
    - meta-llama/Llama-3.2-11B-Vision-Instruct (larger)
    - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo (very capable)
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    - Qwen/Qwen2.5-72B-Instruct-Turbo

    Get API key at: https://api.together.xyz/
    Set env var: TOGETHER_API_KEY=your_key_here
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Together API key required. Set TOGETHER_API_KEY environment variable "
                "or pass api_key parameter. Get key at https://api.together.xyz/"
            )
        self.base_url = "https://api.together.xyz/v1"
        self.history = []

    @property
    def provider_name(self) -> str:
        return "Together AI (Cloud)"

    def chat(self, message: str) -> str:
        """Send a message and get response."""
        self.history.append({"role": "user", "content": message})

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            # Together AI is fast - 30s timeout is generous
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            ai_message = data["choices"][0]["message"]["content"]

            self.history.append({"role": "assistant", "content": ai_message})

            return ai_message

        except requests.exceptions.RequestException as e:
            raise Exception(f"Together AI API error: {str(e)}")

    def reset_history(self):
        """Clear conversation history."""
        self.history = []

    @staticmethod
    def get_available_models() -> List[Dict]:
        """
        Return commonly used Together AI serverless models.

        Note: Only serverless models included - these work without
        dedicated endpoints. See https://api.together.ai/models
        """
        return [
            {
                "name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "size": "3B",
                "description": "Fast, good for rapid experiments"
            },
            {
                "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "size": "8B",
                "description": "Balanced speed and capability"
            },
            {
                "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "size": "70B",
                "description": "Very capable, serverless"
            },
            {
                "name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
                "size": "7B",
                "description": "Balanced performance/speed"
            },
            {
                "name": "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "size": "72B",
                "description": "Strong reasoning, multilingual"
            },
            {
                "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "size": "8x7B",
                "description": "MoE architecture, efficient"
            },
        ]

    @staticmethod
    def is_configured() -> bool:
        """Check if Together AI is configured (API key present)."""
        return bool(os.environ.get("TOGETHER_API_KEY"))


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.history = []

    @property
    def provider_name(self) -> str:
        return "Anthropic (Claude)"

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=self.history
        )

        ai_message = response.content[0].text
        self.history.append({"role": "assistant", "content": ai_message})

        return ai_message

    def reset_history(self):
        self.history = []

    @staticmethod
    def is_configured() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


class OpenAIClient(LLMClient):
    """Client for OpenAI models (GPT-4, etc.)."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.history = []

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=0.7
        )

        ai_message = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": ai_message})

        return ai_message

    def reset_history(self):
        self.history = []

    @staticmethod
    def is_configured() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))


def create_llm_client(provider: str, model: str, **kwargs) -> LLMClient:
    """
    Factory function to create appropriate LLM client.

    Args:
        provider: One of 'ollama', 'together', 'anthropic', 'openai'
        model: Model name/identifier
        **kwargs: Provider-specific parameters (api_key, base_url, etc.)

    Returns:
        Configured LLMClient instance

    Example:
        # Ollama (local)
        client = create_llm_client('ollama', 'llama3.2:latest')

        # Together AI (fast cloud)
        client = create_llm_client('together', 'meta-llama/Llama-3.2-3B-Instruct')

        # Anthropic
        client = create_llm_client('anthropic', 'claude-sonnet-4-20250514')
    """
    if provider == 'ollama':
        return OllamaClient(
            model=model,
            base_url=kwargs.get('base_url', 'http://localhost:11434')
        )

    elif provider == 'together':
        return TogetherClient(
            model=model,
            api_key=kwargs.get('api_key')
        )

    elif provider == 'anthropic':
        return AnthropicClient(
            model=model,
            api_key=kwargs.get('api_key')
        )

    elif provider == 'openai':
        return OpenAIClient(
            model=model,
            api_key=kwargs.get('api_key')
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'together', 'anthropic', or 'openai'")


def get_available_providers() -> List[Dict]:
    """
    Get list of available/configured providers.

    Returns list of dicts with:
    - name: provider identifier
    - display_name: human-readable name
    - available: bool, whether it can be used
    - models: list of available models (if applicable)
    """
    providers = []

    # Ollama (local)
    ollama_running = OllamaClient.is_running()
    ollama_models = OllamaClient.get_available_models() if ollama_running else []
    providers.append({
        "name": "ollama",
        "display_name": "Ollama (Local)",
        "available": ollama_running,
        "models": ollama_models,
        "note": "Privacy-first, runs on your hardware" if ollama_running else "Start with: ollama serve"
    })

    # Together AI (cloud)
    together_configured = TogetherClient.is_configured()
    providers.append({
        "name": "together",
        "display_name": "Together AI (Cloud)",
        "available": together_configured,
        "models": TogetherClient.get_available_models() if together_configured else [],
        "note": "Fast inference (2-5s), reduces latency confound" if together_configured else "Set TOGETHER_API_KEY env var"
    })

    # Anthropic (optional)
    anthropic_configured = AnthropicClient.is_configured()
    providers.append({
        "name": "anthropic",
        "display_name": "Anthropic (Claude)",
        "available": anthropic_configured,
        "models": [{"name": "claude-sonnet-4-20250514", "description": "Claude Sonnet 4"}] if anthropic_configured else [],
        "note": "Claude models" if anthropic_configured else "Set ANTHROPIC_API_KEY env var"
    })

    # OpenAI (optional)
    openai_configured = OpenAIClient.is_configured()
    providers.append({
        "name": "openai",
        "display_name": "OpenAI",
        "available": openai_configured,
        "models": [{"name": "gpt-4", "description": "GPT-4"}] if openai_configured else [],
        "note": "GPT models" if openai_configured else "Set OPENAI_API_KEY env var"
    })

    return providers


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


# Test if run directly
if __name__ == "__main__":
    print("Available LLM Providers:")
    print("=" * 50)

    for provider in get_available_providers():
        status = "✓" if provider["available"] else "✗"
        print(f"\n{status} {provider['display_name']}")
        print(f"  {provider['note']}")

        if provider["available"] and provider["models"]:
            print(f"  Models:")
            for model in provider["models"][:3]:  # Show first 3
                name = model.get("name", model)
                desc = model.get("description", "")
                size = model.get("size", "")
                print(f"    - {name} {f'({size})' if size else ''} {f'- {desc}' if desc else ''}")
