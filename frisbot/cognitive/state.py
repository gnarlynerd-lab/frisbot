"""
Cognitive state management for Frisbot conversational companion.
Adapts DKS companion beliefs from vending machine to conversation.

Core beliefs track internal states relevant to conversation:
- Energy: Capacity for detailed engagement
- Mood: Affective valence
- Curiosity: Drive to explore new topics
- Confidence: Self-assuredness in responses
- Social satiation: How "full" from interaction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import time
import json
from shared.bayesian import BeliefState, BeliefCollection, AdaptiveBeliefSystem


@dataclass
class CognitiveState:
    """
    Manages cognitive state for a conversational AI companion.
    
    Core innovation: Beliefs about internal states (energy, mood, etc.)
    are maintained as Gaussian distributions and updated via Bayesian inference.
    Solvency modulates precision across all beliefs (ASA mechanism).
    """
    
    # Identity
    companion_id: str
    created_at: float = field(default_factory=time.time)
    
    # Core belief system
    belief_system: AdaptiveBeliefSystem = field(default_factory=lambda: AdaptiveBeliefSystem(0.7))
    
    # Topic-level beliefs (grow dynamically)
    topic_beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    
    # User model beliefs
    user_beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    
    # Tracking
    current_topic: Optional[str] = None
    last_interaction_time: Optional[float] = None
    interaction_count: int = 0
    accumulated_surprise: float = 0.0
    messages_since_reflection: int = 0
    
    def __post_init__(self):
        """Initialize default beliefs."""
        # Core self-beliefs about internal states
        self.belief_system.add_belief('energy', mu=0.7, sigma=0.15)
        self.belief_system.add_belief('mood', mu=0.6, sigma=0.20)
        self.belief_system.add_belief('curiosity', mu=0.5, sigma=0.25)
        self.belief_system.add_belief('confidence', mu=0.5, sigma=0.30)
        self.belief_system.add_belief('social_satiation', mu=0.3, sigma=0.20)
        
        # Initialize user model
        self.user_beliefs = {
            'engagement': BeliefState(mu=0.5, sigma=0.30),
            'relationship_depth': BeliefState(mu=0.2, sigma=0.30),
            'preferred_depth': BeliefState(mu=0.5, sigma=0.30),
        }
    
    @property
    def solvency(self) -> float:
        """Current solvency/resource level."""
        return self.belief_system.modulator.solvency
    
    def get_effective_beliefs(self) -> Dict[str, Dict]:
        """
        Get all beliefs with solvency-modulated uncertainty.
        This is what actually drives behavior.
        """
        effective = {}
        for name, belief in self.belief_system.beliefs.beliefs.items():
            effective[name] = self.belief_system.modulator.get_effective_belief(belief)
        return effective
    
    def update_from_message(self, analysis: Dict) -> Dict[str, float]:
        """
        Update beliefs based on analyzed message.
        
        Args:
            analysis: Message analysis with topic, sentiment, novelty, etc.
            
        Returns:
            Dictionary of prediction errors
        """
        errors = {}
        
        # Social satiation increases with interaction
        social_update = min(0.9, self.belief_system.beliefs.get('social_satiation').mu + 0.05)
        errors['social'] = self.belief_system.update_belief(
            'social_satiation', social_update, obs_sigma=0.1
        )
        
        # Mood influenced by sentiment
        if 'sentiment' in analysis:
            mood_influence = 0.6 + (analysis['sentiment'] * 0.2)  # Range: 0.4 to 0.8
            errors['mood'] = self.belief_system.update_belief(
                'mood', mood_influence, obs_sigma=0.2
            )
        
        # Curiosity affected by novelty
        if 'novelty' in analysis and analysis['novelty'] > 0.5:
            # Novel topics increase curiosity
            curiosity_boost = min(0.9, self.belief_system.beliefs.get('curiosity').mu + 0.1)
            errors['curiosity'] = self.belief_system.update_belief(
                'curiosity', curiosity_boost, obs_sigma=0.15
            )
        
        # Update user engagement belief
        if 'engagement_signal' in analysis:
            engagement = BeliefState(mu=analysis['engagement_signal'], sigma=0.2)
            # Simple update for user beliefs (not through main system)
            self.user_beliefs['engagement'] = engagement
        
        # Track topic familiarity
        if 'topic' in analysis:
            self.current_topic = analysis['topic']
            if analysis['topic'] not in self.topic_beliefs:
                # New topic - start uncertain
                self.topic_beliefs[analysis['topic']] = BeliefState(mu=0.3, sigma=0.4)
            else:
                # Increase familiarity with topic
                topic_belief = self.topic_beliefs[analysis['topic']]
                topic_belief.mu = min(1.0, topic_belief.mu + 0.1)
                topic_belief.sigma = max(0.1, topic_belief.sigma * 0.95)  # Reduce uncertainty
        
        # Update surprise accumulation
        total_surprise = sum(abs(e) for e in errors.values() if e is not None)
        self.accumulated_surprise += total_surprise
        self.messages_since_reflection += 1
        
        return errors
    
    def deplete_resources(self, response_length: int, topic_novelty: float = 0.0):
        """
        Deplete resources after generating a response.
        
        Args:
            response_length: Length of generated response
            topic_novelty: How novel/unfamiliar the topic is
        """
        # Calculate complexity based on response length and novelty
        base_complexity = min(1.0, response_length / 2000)  # Normalize to 0-1
        
        # Novel topics require more effort
        if topic_novelty > 0.5:
            base_complexity *= 1.3
        
        # Process the interaction cost
        result = self.belief_system.process_interaction(
            complexity=base_complexity,
            quality=0.0  # No quality gain from our own response
        )
        
        # Energy depletes with effort
        current_energy = self.belief_system.beliefs.get('energy').mu
        new_energy = max(0.1, current_energy - (base_complexity * 0.1))
        self.belief_system.update_belief('energy', new_energy, obs_sigma=0.1)
        
        # Update interaction tracking
        self.last_interaction_time = time.time()
        self.interaction_count += 1
    
    def recover(self, hours_elapsed: float):
        """
        Recovery dynamics between sessions.
        
        Args:
            hours_elapsed: Time since last interaction in hours
        """
        recovery_rate = min(hours_elapsed / 24.0, 1.0)
        
        # Energy recovers
        current_energy = self.belief_system.beliefs.get('energy').mu
        recovered_energy = min(1.0, current_energy + (recovery_rate * 0.3))
        self.belief_system.update_belief('energy', recovered_energy, obs_sigma=0.05)
        
        # Social satiation decays (companion "misses" interaction)
        current_social = self.belief_system.beliefs.get('social_satiation').mu
        decayed_social = max(0.0, current_social - (recovery_rate * 0.2))
        self.belief_system.update_belief('social_satiation', decayed_social, obs_sigma=0.1)
        
        # All beliefs drift toward higher uncertainty
        for name, belief in self.belief_system.beliefs.beliefs.items():
            belief.sigma = min(0.5, belief.sigma * (1.0 + recovery_rate * 0.1))
        
        # Solvency recovers toward target
        self.belief_system.modulator.apply_homeostasis(rate=recovery_rate * 0.2)
        
        # Curiosity refreshes
        current_curiosity = self.belief_system.beliefs.get('curiosity').mu
        refreshed_curiosity = current_curiosity + (recovery_rate * 0.1)
        self.belief_system.update_belief('curiosity', min(0.8, refreshed_curiosity), obs_sigma=0.1)
    
    def should_trigger_reflection(self) -> bool:
        """Check if deep reflection (System 2) should trigger."""
        SURPRISE_THRESHOLD = 2.0
        MAX_MESSAGES = 50
        
        return (
            self.accumulated_surprise > SURPRISE_THRESHOLD or
            self.messages_since_reflection > MAX_MESSAGES
        )
    
    def reset_reflection_counters(self):
        """Reset counters after reflection."""
        self.accumulated_surprise = 0.0
        self.messages_since_reflection = 0
    
    def to_dict(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'companion_id': self.companion_id,
            'created_at': self.created_at,
            'belief_system': self.belief_system.get_system_state(),
            'topic_beliefs': {
                topic: belief.to_dict() 
                for topic, belief in self.topic_beliefs.items()
            },
            'user_beliefs': {
                name: belief.to_dict()
                for name, belief in self.user_beliefs.items()
            },
            'current_topic': self.current_topic,
            'last_interaction_time': self.last_interaction_time,
            'interaction_count': self.interaction_count,
            'accumulated_surprise': self.accumulated_surprise,
            'messages_since_reflection': self.messages_since_reflection,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveState':
        """Restore from serialized state."""
        state = cls(companion_id=data['companion_id'])
        state.created_at = data.get('created_at', time.time())
        
        # Restore belief system
        if 'belief_system' in data:
            bs = data['belief_system']
            # Restore beliefs
            for name, belief_data in bs.get('beliefs', {}).items():
                state.belief_system.beliefs.beliefs[name] = BeliefState.from_dict(belief_data)
            # Restore solvency
            state.belief_system.modulator.solvency = bs.get('solvency', 0.7)
        
        # Restore topic beliefs
        if 'topic_beliefs' in data:
            state.topic_beliefs = {
                topic: BeliefState.from_dict(belief_data)
                for topic, belief_data in data['topic_beliefs'].items()
            }
        
        # Restore user beliefs
        if 'user_beliefs' in data:
            state.user_beliefs = {
                name: BeliefState.from_dict(belief_data)
                for name, belief_data in data['user_beliefs'].items()
            }
        
        # Restore tracking
        state.current_topic = data.get('current_topic')
        state.last_interaction_time = data.get('last_interaction_time')
        state.interaction_count = data.get('interaction_count', 0)
        state.accumulated_surprise = data.get('accumulated_surprise', 0.0)
        state.messages_since_reflection = data.get('messages_since_reflection', 0)
        
        return state
    
    def get_summary(self) -> Dict:
        """Get human-readable summary of current state."""
        effective = self.get_effective_beliefs()
        
        return {
            'solvency': round(self.solvency, 3),
            'energy': {
                'value': round(effective['energy']['mu'], 2),
                'uncertainty': round(effective['energy']['sigma'], 2)
            },
            'mood': {
                'value': round(effective['mood']['mu'], 2),
                'uncertainty': round(effective['mood']['sigma'], 2)
            },
            'curiosity': {
                'value': round(effective['curiosity']['mu'], 2),
                'uncertainty': round(effective['curiosity']['sigma'], 2)
            },
            'social_satiation': {
                'value': round(effective['social_satiation']['mu'], 2),
                'uncertainty': round(effective['social_satiation']['sigma'], 2)
            },
            'confidence': {
                'value': round(effective['confidence']['mu'], 2),
                'uncertainty': round(effective['confidence']['sigma'], 2)
            },
            'current_topic': self.current_topic,
            'interaction_count': self.interaction_count,
            'needs_reflection': self.should_trigger_reflection()
        }