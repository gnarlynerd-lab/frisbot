"""
DeepSeek LLM client for Frisbot.
DeepSeek offers excellent performance at a fraction of the cost.
"""

import os
import json
from typing import List, Dict, Optional, Any
import httpx
from pydantic import BaseModel


class DeepSeekClient:
    """
    Client for DeepSeek API.
    Handles both chat completions and message analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (or from environment)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable.")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Model selection
        self.chat_model = "deepseek-chat"  # Main conversational model
        self.coder_model = "deepseek-coder"  # For analysis tasks (cheaper)
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate a chat response using DeepSeek.
        
        Args:
            messages: Conversation history with system/user/assistant roles
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            Generated response text
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            data = response.json()
            return data['choices'][0]['message']['content']
    
    async def analyze_message(
        self,
        message: str,
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a message for topic, sentiment, novelty, etc.
        Uses the cheaper coder model for this structured task.
        
        Args:
            message: User message to analyze
            context: Recent conversation context
            
        Returns:
            Analysis dictionary
        """
        prompt = f"""Analyze this message and return JSON only, no other text.

Message: "{message}"

Recent context: {context[-3:] if context else "start of conversation"}

Return this exact JSON structure:
{{
    "topic": "primary topic in 1-3 words",
    "sentiment": -1.0 to 1.0 (very negative to very positive),
    "emotional_intensity": 0.0 to 1.0,
    "is_question": true or false,
    "novelty": 0.0 to 1.0 (how different from recent topics),
    "depth": 0.0 to 1.0 (shallow chat vs deep discussion),
    "engagement_signal": 0.0 to 1.0 (how engaged the user seems)
}}

JSON only:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.coder_model,  # Cheaper for structured tasks
                    "messages": [
                        {"role": "system", "content": "You are a message analyzer. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temp for consistent structure
                    "max_tokens": 150
                },
                timeout=20.0
            )
            
            if response.status_code != 200:
                # Fallback to basic analysis if API fails
                return {
                    "topic": "general",
                    "sentiment": 0.0,
                    "emotional_intensity": 0.5,
                    "is_question": "?" in message,
                    "novelty": 0.5,
                    "depth": 0.5,
                    "engagement_signal": 0.5
                }
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            try:
                # Parse JSON response
                analysis = json.loads(content)
                
                # Validate and clamp values
                analysis['sentiment'] = max(-1.0, min(1.0, float(analysis.get('sentiment', 0.0))))
                analysis['emotional_intensity'] = max(0.0, min(1.0, float(analysis.get('emotional_intensity', 0.5))))
                analysis['novelty'] = max(0.0, min(1.0, float(analysis.get('novelty', 0.5))))
                analysis['depth'] = max(0.0, min(1.0, float(analysis.get('depth', 0.5))))
                analysis['engagement_signal'] = max(0.0, min(1.0, float(analysis.get('engagement_signal', 0.5))))
                
                return analysis
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback if parsing fails
                print(f"Failed to parse analysis: {e}")
                return {
                    "topic": "general",
                    "sentiment": 0.0,
                    "emotional_intensity": 0.5,
                    "is_question": "?" in message,
                    "novelty": 0.5,
                    "depth": 0.5,
                    "engagement_signal": 0.5
                }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """
        Estimate cost for API usage.
        
        DeepSeek pricing (as of 2024):
        - deepseek-chat: $0.14/1M input, $0.28/1M output tokens
        - deepseek-coder: $0.14/1M input, $0.28/1M output tokens
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (defaults to chat model)
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.chat_model
        
        # Prices per million tokens
        if model == self.coder_model:
            input_price = 0.14
            output_price = 0.28
        else:  # chat model
            input_price = 0.14
            output_price = 0.28
        
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost