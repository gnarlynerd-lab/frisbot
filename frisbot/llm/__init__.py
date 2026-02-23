"""
LLM integration for Frisbot.
Uses DeepSeek for cost-effective language generation.
"""

from .client import DeepSeekClient
from .prompts import SYSTEM_PROMPT, build_conversation_prompt

__all__ = [
    'DeepSeekClient',
    'SYSTEM_PROMPT',
    'build_conversation_prompt'
]