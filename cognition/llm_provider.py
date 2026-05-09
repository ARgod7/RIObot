"""
LLM Provider Wrapper — Unified interface for Groq, Gemini, and Ollama.

This module provides a single factory function that returns the configured LLM.
Supports all three providers with sensible defaults for RIO's dialogue and reasoning tasks.
"""

import os
import logging

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


def get_llm():
    """Return a CrewAI LLM instance for the configured provider.

    Uses the CrewAI LLM wrapper with LiteLLM backend.
    Supports: Groq, Gemini, and Ollama.

    Returns:
        crewai.LLM instance configured with appropriate model and API key.

    Raises:
        ValueError: If LLM_PROVIDER is not recognized.
    """
    try:
        from crewai import LLM
    except ImportError:
        raise ImportError("crewai not installed. Run: pip install crewai")

    provider = LLM_PROVIDER.lower()

    if provider == "groq":
        logger.info("Using Groq LLM: groq/llama-3.3-70b-versatile")
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    elif provider == "ollama":
        logger.info("Using Ollama LLM: ollama/llama3.2")
        return LLM(
            model="ollama/llama3.2",
            api_key=os.getenv("OLLAMA_API_KEY")
        )

    elif provider == "gemini":
        logger.info("Using Gemini LLM: gemini/gemini-1.5-flash")
        return LLM(
            model="gemini/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Supported: 'groq', 'gemini', 'ollama'"
        )

