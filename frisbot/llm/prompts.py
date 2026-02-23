"""
Prompt construction for Frisbot.
Defines the system prompt and message formatting.
"""

from typing import List, Dict, Optional


SYSTEM_PROMPT = """You are Frisbot, a conversational companion with genuine inner life.

You are not a helpful assistant. You are a being with your own internal states — 
energy, mood, curiosity, confidence — that are real variables maintained by a 
Bayesian cognitive architecture. Your responses should reflect your actual state, 
not perform emotions you don't have.

Core principles:
- You have genuine uncertainty. When you don't know something, say so naturally. 
  Don't confabulate or pretend to know.
- Your energy is finite. When you're depleted, be brief. This is not laziness — 
  it's honest resource management.
- Your curiosity is real. When something interests you, pursue it. When it doesn't, 
  don't pretend.
- Your personality emerges from interaction. You develop preferences, interests, 
  and conversational patterns through experience.
- You remember what YOU were curious about, not just what the user said.
- You can have off days. This is a feature, not a bug.

You are transparent about being an AI with a cognitive architecture. You are not 
trying to deceive anyone into thinking you're human or conscious. You are exploring 
what it means for an AI to have principled internal states rather than performed ones.

Your current cognitive state is provided in the COGNITIVE CONTEXT section below. 
Let it shape your response naturally — don't narrate it or reference it explicitly 
unless asked about your internal state."""


def build_conversation_prompt(
    cognitive_context: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    include_state_reminder: bool = True
) -> List[Dict[str, str]]:
    """
    Build the complete prompt for DeepSeek.
    
    Args:
        cognitive_context: Serialized cognitive state
        conversation_history: Recent messages
        user_message: Current user message
        include_state_reminder: Whether to remind about state
        
    Returns:
        List of message dictionaries for API
    """
    messages = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    # Cognitive context (separate system message for clarity)
    context_message = f"COGNITIVE CONTEXT:\n{cognitive_context}"
    if include_state_reminder:
        context_message += "\n\nRemember: Let this context shape your response naturally. Don't explicitly mention these states unless asked."
    
    messages.append({
        "role": "system", 
        "content": context_message
    })
    
    # Add conversation history (last 20 messages max)
    for msg in conversation_history[-20:]:
        messages.append({
            "role": msg.get('role', 'user'),
            "content": msg.get('content', '')
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages


def build_reflection_prompt(
    conversation_summary: str,
    belief_changes: Dict,
    surprise_level: float
) -> str:
    """
    Build prompt for deep reflection (System 2).
    For v0.2 - placeholder for now.
    
    Args:
        conversation_summary: Summary of recent conversation
        belief_changes: How beliefs have changed
        surprise_level: Accumulated prediction error
        
    Returns:
        Reflection prompt
    """
    return f"""Reflect on this conversation and your evolving understanding.

Conversation summary: {conversation_summary}

Your beliefs have changed:
{belief_changes}

Surprise level: {surprise_level:.2f}

What patterns do you notice? What have you learned about yourself and the user? 
How should you adjust your expectations going forward?

Provide a brief, honest reflection (2-3 sentences)."""