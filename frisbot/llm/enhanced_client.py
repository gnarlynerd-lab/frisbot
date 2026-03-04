"""
Enhanced DeepSeek LLM client with engagement quality evaluation.
DeepSeek evaluates the quality of user engagement to inform solvency adjustments.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
import httpx
from pydantic import BaseModel


class EnhancedDeepSeekClient:
    """
    Enhanced client for DeepSeek API.
    Includes engagement quality evaluation for solvency-based adaptation.
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
    
    async def analyze_message_with_quality(
        self,
        message: str,
        context: Optional[List[str]] = None,
        previous_state: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Analyze a message for topic, sentiment, novelty, and engagement quality.
        DeepSeek evaluates the quality of the interaction to inform solvency.
        
        Args:
            message: User message to analyze
            context: Recent conversation context
            previous_state: Previous cognitive state for quality assessment
            
        Returns:
            Tuple of (analysis dictionary, engagement quality score)
        """
        state_context = ""
        if previous_state:
            state_context = f"""
Current companion state:
- Energy: {previous_state.get('energy', {}).get('value', 0.5):.2f}
- Mood: {previous_state.get('mood', {}).get('value', 0.5):.2f}
- Curiosity: {previous_state.get('curiosity', {}).get('value', 0.5):.2f}
- Social Satiation: {previous_state.get('social_satiation', {}).get('value', 0.3):.2f}
- Solvency: {previous_state.get('solvency', 0.7):.2f}
"""
        
        prompt = f"""Analyze this message and evaluate the quality of engagement.
Consider both the message content and how well it engages with the companion's current state.

Message: "{message}"

Recent context: {context[-3:] if context else "start of conversation"}
{state_context}

Return this exact JSON structure:
{{
    "topic": "primary topic in 1-3 words",
    "sentiment": -1.0 to 1.0 (very negative to very positive),
    "emotional_intensity": 0.0 to 1.0,
    "is_question": true or false,
    "novelty": 0.0 to 1.0 (how different from recent topics),
    "depth": 0.0 to 1.0 (shallow chat vs deep discussion),
    "engagement_signal": 0.0 to 1.0 (how engaged the user seems),
    "quality_assessment": {{
        "relevance": 0.0 to 1.0 (how relevant to conversation flow),
        "thoughtfulness": 0.0 to 1.0 (how considered/thoughtful),
        "reciprocity": 0.0 to 1.0 (builds on previous exchange),
        "authenticity": 0.0 to 1.0 (genuine vs perfunctory),
        "enrichment": 0.0 to 1.0 (adds value to conversation),
        "overall_quality": 0.0 to 1.0 (weighted assessment)
    }},
    "solvency_impact": {{
        "suggested_adjustment": -0.2 to 0.2 (how much to adjust solvency),
        "reasoning": "brief explanation of why"
    }}
}}

JSON only:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.coder_model,  # Cheaper for structured tasks
                    "messages": [
                        {"role": "system", "content": "You are an engagement quality analyzer. Evaluate how well users engage with an AI companion that has limited cognitive resources (solvency). High-quality, thoughtful engagement should restore resources, while low-quality or demanding interactions should deplete them. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temp for consistent structure
                    "max_tokens": 300
                },
                timeout=20.0
            )
            
            if response.status_code != 200:
                # Fallback to basic analysis if API fails
                return self._fallback_analysis(message), 0.0
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            try:
                # Parse JSON response
                analysis = json.loads(content)
                
                # Validate and clamp basic values
                analysis['sentiment'] = max(-1.0, min(1.0, float(analysis.get('sentiment', 0.0))))
                analysis['emotional_intensity'] = max(0.0, min(1.0, float(analysis.get('emotional_intensity', 0.5))))
                analysis['novelty'] = max(0.0, min(1.0, float(analysis.get('novelty', 0.5))))
                analysis['depth'] = max(0.0, min(1.0, float(analysis.get('depth', 0.5))))
                analysis['engagement_signal'] = max(0.0, min(1.0, float(analysis.get('engagement_signal', 0.5))))
                
                # Extract quality assessment
                quality_assessment = analysis.get('quality_assessment', {})
                overall_quality = float(quality_assessment.get('overall_quality', 0.5))
                
                # Calculate engagement quality score for solvency adjustment
                # This combines multiple factors with weights
                quality_score = self._calculate_quality_score(analysis, quality_assessment)
                
                return analysis, quality_score
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback if parsing fails
                print(f"Failed to parse analysis: {e}")
                return self._fallback_analysis(message), 0.0
    
    def _calculate_quality_score(self, analysis: Dict, quality_assessment: Dict) -> float:
        """
        Calculate a unified quality score for solvency adjustment.
        
        High quality interactions (score > 0) should increase solvency.
        Low quality interactions (score < 0) should decrease solvency.
        
        Args:
            analysis: Full message analysis
            quality_assessment: Quality-specific metrics
            
        Returns:
            Quality score between -1.0 and 1.0
        """
        # Extract quality metrics with defaults
        relevance = float(quality_assessment.get('relevance', 0.5))
        thoughtfulness = float(quality_assessment.get('thoughtfulness', 0.5))
        reciprocity = float(quality_assessment.get('reciprocity', 0.5))
        authenticity = float(quality_assessment.get('authenticity', 0.5))
        enrichment = float(quality_assessment.get('enrichment', 0.5))
        
        # Extract engagement metrics
        depth = float(analysis.get('depth', 0.5))
        engagement = float(analysis.get('engagement_signal', 0.5))
        
        # Weighted combination
        quality_components = [
            (relevance, 0.15),      # Is this on-topic?
            (thoughtfulness, 0.20),  # Is effort evident?
            (reciprocity, 0.20),     # Does it build on conversation?
            (authenticity, 0.15),    # Feels genuine?
            (enrichment, 0.10),      # Adds value?
            (depth, 0.10),           # Substantive content?
            (engagement, 0.10)       # User engaged?
        ]
        
        weighted_sum = sum(value * weight for value, weight in quality_components)
        
        # Center around 0 (0.5 becomes 0, 1.0 becomes 1.0, 0.0 becomes -1.0)
        centered_score = (weighted_sum - 0.5) * 2
        
        # Apply non-linearity: reward high quality more, penalize low quality less
        if centered_score > 0:
            # Amplify positive scores slightly
            final_score = centered_score ** 0.9
        else:
            # Dampen negative scores slightly  
            final_score = -(abs(centered_score) ** 1.1)
        
        return max(-1.0, min(1.0, final_score))
    
    def _fallback_analysis(self, message: str) -> Dict:
        """Fallback analysis when API fails."""
        return {
            "topic": "general",
            "sentiment": 0.0,
            "emotional_intensity": 0.5,
            "is_question": "?" in message,
            "novelty": 0.5,
            "depth": 0.5,
            "engagement_signal": 0.5,
            "quality_assessment": {
                "relevance": 0.5,
                "thoughtfulness": 0.5,
                "reciprocity": 0.5,
                "authenticity": 0.5,
                "enrichment": 0.5,
                "overall_quality": 0.5
            },
            "solvency_impact": {
                "suggested_adjustment": 0.0,
                "reasoning": "Unable to assess"
            }
        }
    
    async def evaluate_response_quality(
        self,
        user_message: str,
        assistant_response: str,
        post_state: Dict
    ) -> float:
        """
        Evaluate how well the assistant's response addressed the user's needs.
        This helps calibrate future solvency adjustments.
        
        Args:
            user_message: The user's original message
            assistant_response: The assistant's response
            post_state: Cognitive state after response
            
        Returns:
            Response quality score (-1.0 to 1.0)
        """
        prompt = f"""Evaluate how well this assistant response addressed the user's message.

User message: "{user_message}"

Assistant response: "{assistant_response}"

Assistant's post-response state:
- Solvency: {post_state.get('solvency', 0.7):.2f}
- Energy: {post_state.get('energy', {}).get('value', 0.5):.2f}

Rate the response quality:
{{
    "addressed_question": 0.0 to 1.0 (if question, was it answered?),
    "emotional_resonance": 0.0 to 1.0 (appropriate emotional tone?),
    "coherence": 0.0 to 1.0 (makes sense in context?),
    "value_added": 0.0 to 1.0 (provides value to user?),
    "overall_effectiveness": -1.0 to 1.0 (centered score)
}}

JSON only:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.coder_model,
                    "messages": [
                        {"role": "system", "content": "You are a response quality evaluator. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150
                },
                timeout=20.0
            )
            
            if response.status_code != 200:
                return 0.0
            
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                evaluation = json.loads(content)
                return float(evaluation.get('overall_effectiveness', 0.0))
            except:
                return 0.0
    
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