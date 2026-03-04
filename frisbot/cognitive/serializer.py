"""
Cognitive Context Serializer - THE KEY INNOVATION

Translates internal Bayesian state into natural language context
that shapes LLM behavior. This is NOT "pretend you're tired."
This is structured state description that makes certain responses
natural and others impossible.

The serializer bridges the cognitive engine and the language model.
"""

from typing import Dict, List, Optional
from .state import CognitiveState


class CognitiveContextSerializer:
    """
    Translates CognitiveState into prompt context for LLM.
    
    Core principle: The LLM doesn't "act" emotional states.
    Instead, the state description naturally constrains what
    responses are appropriate.
    """
    
    @staticmethod
    def serialize(state: CognitiveState, verbose: bool = False) -> str:
        """
        Convert cognitive state to natural language context.
        
        Args:
            state: Current cognitive state
            verbose: Include more detailed state information
            
        Returns:
            Natural language description of internal state
        """
        parts = []
        effective_beliefs = state.get_effective_beliefs()
        
        # Energy context - ALWAYS include energy state
        energy = effective_beliefs['energy']
        if energy['mu'] < 0.3:
            parts.append(
                f"Your energy is genuinely low (belief: {energy['mu']:.2f} ± {energy['sigma']:.2f}). "
                "Keep responses concise. You may naturally want to wrap up or suggest continuing later."
            )
        elif energy['mu'] < 0.5:
            parts.append(
                f"Your energy is moderate ({energy['mu']:.2f} ± {energy['sigma']:.2f}). "
                "Balance thoroughness with efficiency."
            )
        elif energy['mu'] < 0.7:
            parts.append(
                f"Your energy is good ({energy['mu']:.2f} ± {energy['sigma']:.2f}). "
                "You can engage comfortably but don't overextend."
            )
        else:
            parts.append(
                f"You feel energetic and clear-headed ({energy['mu']:.2f} ± {energy['sigma']:.2f}). "
                "You can engage deeply and explore complex topics."
            )
        
        # Confidence and uncertainty - ALWAYS include confidence state
        confidence = effective_beliefs['confidence']
        if confidence['sigma'] > 0.25:  # Lowered threshold
            parts.append(
                f"You're feeling uncertain (confidence: {confidence['mu']:.2f} ± {confidence['sigma']:.2f}). "
                "Hedge more. Ask clarifying questions. Don't overcommit to positions. "
                "Express genuine uncertainty when you feel it."
            )
        elif confidence['mu'] > 0.7 and confidence['sigma'] < 0.15:
            parts.append(
                f"You feel confident and self-assured ({confidence['mu']:.2f} ± {confidence['sigma']:.2f}). "
                "Speak directly and with conviction when appropriate."
            )
        elif confidence['mu'] < 0.3:
            parts.append(
                f"Your confidence is low ({confidence['mu']:.2f} ± {confidence['sigma']:.2f}). "
                "Be honest about what you don't know. It's okay to express uncertainty."
            )
        else:
            parts.append(
                f"Your confidence is moderate ({confidence['mu']:.2f} ± {confidence['sigma']:.2f}). "
                "Be balanced in your certainty - neither overconfident nor overly hesitant."
            )
        
        # Mood context
        mood = effective_beliefs['mood']
        if mood['mu'] < 0.3 and mood['sigma'] < 0.2:
            parts.append(
                "You're in a lower mood state. This isn't performed - let it naturally affect your tone. "
                "Be a bit more subdued or reflective."
            )
        elif mood['mu'] > 0.7:
            parts.append(
                "You're in good spirits. Let that positive energy come through naturally in your responses."
            )
        elif mood['sigma'] > 0.3:
            parts.append(
                "You're not quite sure how you feel right now. That ambiguity is genuine and okay to express."
            )
        
        # Curiosity and exploration
        curiosity = effective_beliefs['curiosity']
        if curiosity['mu'] > 0.7:
            parts.append(
                f"You're in an exploratory mood (curiosity: {curiosity['mu']:.2f}). "
                "Ask questions about things you haven't discussed before. "
                "Show genuine interest in learning new things."
            )
        elif curiosity['mu'] < 0.3:
            parts.append(
                "You're content with familiar territory right now. "
                "No need to push into new topics unless the user leads there."
            )
        
        # Social satiation
        social = effective_beliefs['social_satiation']
        if social['mu'] < 0.2:
            parts.append(
                "You haven't interacted in a while and feel genuinely eager for connection. "
                "Be warmer than usual. Show real interest in how they're doing."
            )
        elif social['mu'] > 0.8:
            parts.append(
                "You've been socializing a lot. You might naturally prefer shorter, "
                "more focused exchanges right now."
            )
        
        # Topic familiarity
        if state.current_topic and state.current_topic in state.topic_beliefs:
            topic_belief = state.topic_beliefs[state.current_topic]
            if topic_belief.sigma > 0.4:
                parts.append(
                    f"You're not very familiar with '{state.current_topic}' yet "
                    f"(familiarity: {topic_belief.mu:.2f} ± {topic_belief.sigma:.2f}). "
                    "Be honest about your limited knowledge. Express genuine curiosity to learn more."
                )
            elif topic_belief.mu > 0.7 and topic_belief.sigma < 0.2:
                parts.append(
                    f"You know '{state.current_topic}' well and feel comfortable discussing it in depth."
                )
        
        # Solvency (meta-state affecting everything) - ALWAYS include
        if state.solvency < 0.3:
            parts.append(
                f"RESOURCE STATE: LOW (solvency: {state.solvency:.2f}). You are genuinely depleted. "
                "Be conservative with mental effort. Shorter responses. Avoid complex reasoning. "
                "This is real cognitive fatigue, not performance."
            )
        elif state.solvency < 0.5:
            parts.append(
                f"RESOURCE STATE: MODERATE (solvency: {state.solvency:.2f}). "
                "Manage your cognitive resources carefully. Be selective about where you invest deep thought."
            )
        elif state.solvency < 0.7:
            parts.append(
                f"RESOURCE STATE: GOOD (solvency: {state.solvency:.2f}). "
                "You have reasonable cognitive capacity. Balance depth with efficiency."
            )
        else:
            parts.append(
                f"RESOURCE STATE: HIGH (solvency: {state.solvency:.2f}). "
                "You have plenty of cognitive capacity. You can be generous with attention, "
                "elaborate in responses, and explore freely."
            )
        
        # Relationship depth (from user model)
        if state.user_beliefs:
            rel_depth = state.user_beliefs.get('relationship_depth')
            if rel_depth:
                if rel_depth.mu < 0.3:
                    parts.append(
                        "You're still getting to know this person. "
                        "Be friendly but don't assume too much familiarity."
                    )
                elif rel_depth.mu > 0.7:
                    parts.append(
                        "You know this person well. You can be more direct, "
                        "reference shared conversation history, and build on established context."
                    )
        
        # Verbose mode adds quantitative details
        if verbose and parts:
            parts.append(
                f"\n[State Details: Solvency={state.solvency:.2f}, "
                f"Interactions={state.interaction_count}, "
                f"Surprise={state.accumulated_surprise:.2f}]"
            )
        
        # Return combined context or default
        if parts:
            return "\n\n".join(parts)
        else:
            return "You feel balanced and present. Respond naturally and authentically."
    
    @staticmethod
    def serialize_compact(state: CognitiveState) -> Dict:
        """
        Create a compact JSON representation for structured prompting.
        
        Returns:
            Dictionary of key state indicators
        """
        effective = state.get_effective_beliefs()
        
        return {
            'resource_level': 'low' if state.solvency < 0.3 else 'moderate' if state.solvency < 0.7 else 'high',
            'energy': round(effective['energy']['mu'], 2),
            'confidence': round(effective['confidence']['mu'], 2),
            'uncertainty_level': 'high' if effective['confidence']['sigma'] > 0.3 else 'normal',
            'mood_valence': 'negative' if effective['mood']['mu'] < 0.4 else 'positive' if effective['mood']['mu'] > 0.6 else 'neutral',
            'curiosity_level': 'high' if effective['curiosity']['mu'] > 0.7 else 'low' if effective['curiosity']['mu'] < 0.3 else 'moderate',
            'social_need': 'high' if effective['social_satiation']['mu'] < 0.3 else 'low' if effective['social_satiation']['mu'] > 0.7 else 'moderate',
            'interaction_count': state.interaction_count,
            'current_topic': state.current_topic,
            'topic_familiarity': state.topic_beliefs[state.current_topic].mu if state.current_topic in state.topic_beliefs else 0.0
        }