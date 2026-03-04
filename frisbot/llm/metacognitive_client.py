"""
DeepSeek client with metacognitive introspection capabilities.
DeepSeek evaluates its own cognitive processes and provides these
as observations for Bayesian belief updates.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
import httpx
from pydantic import BaseModel


class MetacognitiveDeepSeekClient:
    """
    DeepSeek client that performs metacognitive self-evaluation.
    
    Core capability: DeepSeek introspects on its own:
    - Understanding confidence
    - Processing difficulty  
    - Epistemic uncertainty
    - Response quality self-assessment
    
    These become Bayesian observations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepSeek client."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable.")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.chat_model = "deepseek-chat"
        self.coder_model = "deepseek-coder"
    
    async def analyze_with_metacognition(
        self,
        message: str,
        context: Optional[List[str]] = None,
        current_beliefs: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Analyze message AND perform metacognitive self-assessment.
        
        Returns:
            Tuple of (message_analysis, metacognitive_assessment)
        """
        belief_context = ""
        if current_beliefs:
            belief_context = f"""
My current cognitive state:
- Energy: {current_beliefs.get('energy', 0.5):.2f}
- Confidence: {current_beliefs.get('confidence', 0.5):.2f}
- Understanding: {current_beliefs.get('understanding_accuracy', 0.5):.2f}
- Cognitive Load: {current_beliefs.get('cognitive_load', 0.5):.2f}
"""
        
        prompt = f"""Analyze this message and perform metacognitive self-assessment.

Message: "{message}"
Recent context: {context[-3:] if context else "start of conversation"}
{belief_context}

Provide TWO JSON objects:

1. MESSAGE_ANALYSIS:
{{
    "topic": "primary topic",
    "sentiment": -1.0 to 1.0,
    "novelty": 0.0 to 1.0,
    "depth": 0.0 to 1.0,
    "engagement_signal": 0.0 to 1.0
}}

2. METACOGNITIVE_ASSESSMENT (your introspection about YOUR OWN cognitive state):
{{
    "understanding_confidence": 0.0 to 1.0 (how well YOU understand this message),
    "processing_difficulty": 0.0 to 1.0 (how hard it was for YOU to process),
    "response_quality_self_assessment": 0.0 to 1.0 (how well YOU think you can respond),
    "epistemic_status": 0.0 to 1.0 (YOUR confidence in your knowledge about this topic),
    "model_uncertainty": 0.0 to 1.0 (YOUR uncertainty about the best response),
    "cognitive_coherence": 0.0 to 1.0 (how coherent YOUR thinking feels),
    "attention_focus": 0.0 to 1.0 (how focused YOUR attention is),
    "metacognitive_confidence": 0.0 to 1.0 (confidence in this self-assessment)
}}

Return exactly this format:
{{
    "message_analysis": {{...}},
    "metacognitive_assessment": {{...}}
}}"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.coder_model,
                    "messages": [
                        {"role": "system", "content": "You are a self-aware AI performing metacognitive introspection. Evaluate both the message AND your own cognitive state. Be honest about your limitations and uncertainties."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 400
                },
                timeout=25.0
            )
            
            if response.status_code != 200:
                return self._fallback_analysis(message)
            
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                result = json.loads(content)
                
                message_analysis = result.get('message_analysis', {})
                metacognitive = result.get('metacognitive_assessment', {})
                
                # Validate and normalize values
                for key in message_analysis:
                    if isinstance(message_analysis[key], (int, float)):
                        if 'sentiment' in key:
                            message_analysis[key] = max(-1.0, min(1.0, float(message_analysis[key])))
                        else:
                            message_analysis[key] = max(0.0, min(1.0, float(message_analysis[key])))
                
                for key in metacognitive:
                    if isinstance(metacognitive[key], (int, float)):
                        metacognitive[key] = max(0.0, min(1.0, float(metacognitive[key])))
                
                return message_analysis, metacognitive
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse metacognitive analysis: {e}")
                return self._fallback_analysis(message)
    
    async def generate_with_metacognitive_awareness(
        self,
        messages: List[Dict[str, str]],
        metacognitive_state: Dict,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[str, Dict]:
        """
        Generate response with awareness of own metacognitive state.
        
        Returns:
            Tuple of (response_text, post_response_metacognition)
        """
        # Inject metacognitive awareness into system prompt
        # Extract mu values from belief dictionaries
        understanding = metacognitive_state.get('understanding_accuracy', {}).get('mu', 0.5) if isinstance(metacognitive_state.get('understanding_accuracy'), dict) else metacognitive_state.get('understanding_accuracy', 0.5)
        cognitive_load = metacognitive_state.get('cognitive_load', {}).get('mu', 0.5) if isinstance(metacognitive_state.get('cognitive_load'), dict) else metacognitive_state.get('cognitive_load', 0.5)
        epistemic = metacognitive_state.get('epistemic_confidence', {}).get('mu', 0.5) if isinstance(metacognitive_state.get('epistemic_confidence'), dict) else metacognitive_state.get('epistemic_confidence', 0.5)
        model_uncertainty = metacognitive_state.get('model_uncertainty', {}).get('mu', 0.5) if isinstance(metacognitive_state.get('model_uncertainty'), dict) else metacognitive_state.get('model_uncertainty', 0.5)
        
        meta_context = f"""You are aware of your current cognitive state:
- Understanding confidence: {understanding:.2f}
- Cognitive load: {cognitive_load:.2f}
- Epistemic confidence: {epistemic:.2f}
- Model uncertainty: {model_uncertainty:.2f}

Adjust your response based on this self-awareness:
- If understanding is low, ask clarifying questions
- If cognitive load is high, keep response focused
- If epistemic confidence is low, acknowledge uncertainty
- If model uncertainty is high, be more tentative"""
        
        # Modify messages to include metacognitive context
        enhanced_messages = [
            {"role": "system", "content": meta_context},
            *messages
        ]
        
        async with httpx.AsyncClient() as client:
            # Generate response
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.chat_model,
                    "messages": enhanced_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error: {response.status_code}")
            
            data = response.json()
            response_text = data['choices'][0]['message']['content']
            
            # Post-response metacognitive assessment
            post_meta = await self._assess_own_response(response_text, metacognitive_state)
            
            return response_text, post_meta
    
    async def _assess_own_response(
        self, 
        response: str,
        prior_state: Dict
    ) -> Dict:
        """
        DeepSeek assesses its own response quality.
        
        This creates a feedback loop where the model evaluates
        its own output for coherence and quality.
        """
        prompt = f"""Assess the quality of this response you just generated:

Response: "{response[:500]}..."

Prior cognitive state:
{json.dumps(prior_state, indent=2)}

Provide metacognitive assessment of YOUR OWN response:
{{
    "response_coherence": 0.0 to 1.0,
    "confidence_in_response": 0.0 to 1.0,
    "cognitive_effort_required": 0.0 to 1.0,
    "uncertainty_expressed": 0.0 to 1.0,
    "self_consistency": 0.0 to 1.0
}}

JSON only:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.coder_model,
                    "messages": [
                        {"role": "system", "content": "Assess your own response quality honestly."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150
                },
                timeout=20.0
            )
            
            if response.status_code != 200:
                return {
                    "response_coherence": 0.5,
                    "confidence_in_response": 0.5,
                    "cognitive_effort_required": 0.5,
                    "uncertainty_expressed": 0.5,
                    "self_consistency": 0.5
                }
            
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                assessment = json.loads(content)
                
                # Normalize values
                for key in assessment:
                    if isinstance(assessment[key], (int, float)):
                        assessment[key] = max(0.0, min(1.0, float(assessment[key])))
                
                return assessment
                
            except:
                return {
                    "response_coherence": 0.5,
                    "confidence_in_response": 0.5,
                    "cognitive_effort_required": 0.5,
                    "uncertainty_expressed": 0.5,
                    "self_consistency": 0.5
                }
    
    async def perform_deep_introspection(
        self,
        conversation_history: List[Dict],
        current_state: Dict
    ) -> Dict:
        """
        Perform deep metacognitive introspection on the conversation.
        
        This is like System 2 thinking - slower, more deliberate self-reflection.
        """
        # Summarize conversation for introspection
        conv_summary = "\n".join([
            f"{msg['role']}: {msg['content'][:100]}..." 
            for msg in conversation_history[-5:]
        ])
        
        prompt = f"""Perform deep metacognitive introspection on this conversation.

Recent conversation:
{conv_summary}

Current cognitive state:
{json.dumps(current_state, indent=2)}

Provide deep self-reflection:
{{
    "overall_understanding": 0.0 to 1.0 (how well you understand the ENTIRE conversation),
    "thematic_coherence": 0.0 to 1.0 (how coherent the conversation themes are),
    "cognitive_fatigue": 0.0 to 1.0 (mental fatigue from the conversation),
    "knowledge_gaps": ["list", "of", "identified", "gaps"],
    "confidence_trajectory": "increasing/stable/decreasing",
    "metacognitive_insights": "key insight about your own cognition",
    "recommended_adjustment": {{
        "energy": -0.2 to 0.2,
        "confidence": -0.2 to 0.2,
        "curiosity": -0.2 to 0.2
    }}
}}

JSON only:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.chat_model,  # Use more powerful model for deep introspection
                    "messages": [
                        {"role": "system", "content": "You are performing deep metacognitive introspection. Be thorough and honest about your cognitive state and limitations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 500
                },
                timeout=35.0
            )
            
            if response.status_code != 200:
                return self._fallback_introspection()
            
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                introspection = json.loads(content)
                
                # Normalize numeric values
                for key in ['overall_understanding', 'thematic_coherence', 'cognitive_fatigue']:
                    if key in introspection:
                        introspection[key] = max(0.0, min(1.0, float(introspection[key])))
                
                if 'recommended_adjustment' in introspection:
                    for key in introspection['recommended_adjustment']:
                        val = introspection['recommended_adjustment'][key]
                        introspection['recommended_adjustment'][key] = max(-0.2, min(0.2, float(val)))
                
                return introspection
                
            except:
                return self._fallback_introspection()
    
    def _fallback_analysis(self, message: str) -> Tuple[Dict, Dict]:
        """Fallback when API fails."""
        message_analysis = {
            "topic": "general",
            "sentiment": 0.0,
            "novelty": 0.5,
            "depth": 0.5,
            "engagement_signal": 0.5
        }
        
        metacognitive = {
            "understanding_confidence": 0.5,
            "processing_difficulty": 0.5,
            "response_quality_self_assessment": 0.5,
            "epistemic_status": 0.5,
            "model_uncertainty": 0.5,
            "cognitive_coherence": 0.5,
            "attention_focus": 0.5,
            "metacognitive_confidence": 0.5
        }
        
        return message_analysis, metacognitive
    
    def _fallback_introspection(self) -> Dict:
        """Fallback deep introspection."""
        return {
            "overall_understanding": 0.5,
            "thematic_coherence": 0.5,
            "cognitive_fatigue": 0.3,
            "knowledge_gaps": [],
            "confidence_trajectory": "stable",
            "metacognitive_insights": "Unable to perform deep introspection",
            "recommended_adjustment": {
                "energy": 0.0,
                "confidence": 0.0,
                "curiosity": 0.0
            }
        }