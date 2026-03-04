"""
Enhanced cognitive state management with quality-based solvency adjustment.
Integrates DeepSeek's engagement quality evaluation to modulate resources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import json
import numpy as np
from shared.bayesian import BeliefState, BeliefCollection, AdaptiveBeliefSystem


@dataclass
class QualityAdjustedCognitiveState:
    """
    Enhanced cognitive state that adjusts solvency based on engagement quality.
    
    Core innovation: DeepSeek evaluates interaction quality, and this evaluation
    directly modulates the companion's cognitive resources (solvency).
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
    
    # Quality tracking
    quality_history: List[float] = field(default_factory=list)
    cumulative_quality: float = 0.0
    quality_momentum: float = 0.0  # Tracks trend in quality
    
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
        
        # Initialize user model with quality-aware beliefs
        self.user_beliefs = {
            'engagement': BeliefState(mu=0.5, sigma=0.30),
            'relationship_depth': BeliefState(mu=0.2, sigma=0.30),
            'preferred_depth': BeliefState(mu=0.5, sigma=0.30),
            'interaction_quality': BeliefState(mu=0.5, sigma=0.30),  # New: tracks overall quality
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
    
    def update_from_quality_analysis(
        self, 
        analysis: Dict,
        quality_score: float
    ) -> Tuple[Dict[str, float], float]:
        """
        Update beliefs based on analyzed message with quality assessment.
        
        Args:
            analysis: Message analysis with topic, sentiment, novelty, etc.
            quality_score: Engagement quality score from DeepSeek (-1.0 to 1.0)
            
        Returns:
            Tuple of (prediction errors dict, solvency adjustment)
        """
        errors = {}
        
        # Track quality history (keep last 10)
        self.quality_history.append(quality_score)
        if len(self.quality_history) > 10:
            self.quality_history.pop(0)
        
        # Calculate quality momentum (trend)
        if len(self.quality_history) >= 3:
            recent_quality = np.mean(self.quality_history[-3:])
            older_quality = np.mean(self.quality_history[:-3]) if len(self.quality_history) > 3 else 0.5
            self.quality_momentum = recent_quality - older_quality
        
        # Update cumulative quality
        self.cumulative_quality += quality_score
        
        # === QUALITY-BASED SOLVENCY ADJUSTMENT ===
        # This is the key innovation: quality directly affects resources
        
        # Base adjustment from quality score
        base_adjustment = quality_score * 0.15  # Max ±15% per interaction
        
        # Momentum bonus/penalty (trending better or worse)
        momentum_adjustment = self.quality_momentum * 0.05
        
        # Fatigue factor (long conversations with low quality are more draining)
        fatigue_factor = 1.0
        if self.interaction_count > 10 and np.mean(self.quality_history) < 0.3:
            fatigue_factor = 1.5  # Low quality sustained interactions are extra draining
        
        # Calculate final solvency adjustment
        solvency_adjustment = (base_adjustment + momentum_adjustment) / fatigue_factor
        
        # Apply solvency adjustment with bounds
        new_solvency = max(0.1, min(1.0, self.solvency + solvency_adjustment))
        self.belief_system.modulator.solvency = new_solvency
        
        # === STANDARD BELIEF UPDATES ===
        
        # Social satiation increases with interaction (but modulated by quality)
        social_increment = 0.05 if quality_score > 0 else 0.08  # Low quality is less satisfying
        social_update = min(0.9, self.belief_system.beliefs.get('social_satiation').mu + social_increment)
        errors['social'] = self.belief_system.update_belief(
            'social_satiation', social_update, obs_sigma=0.1
        )
        
        # Mood influenced by sentiment AND quality
        if 'sentiment' in analysis:
            mood_base = 0.6 + (analysis['sentiment'] * 0.2)
            mood_quality_boost = quality_score * 0.1  # Quality affects mood
            mood_influence = max(0.2, min(0.9, mood_base + mood_quality_boost))
            errors['mood'] = self.belief_system.update_belief(
                'mood', mood_influence, obs_sigma=0.2
            )
        
        # Curiosity affected by novelty (boosted by high quality novel content)
        if 'novelty' in analysis and analysis['novelty'] > 0.5:
            curiosity_boost = 0.1 if quality_score > 0 else 0.05
            new_curiosity = min(0.9, self.belief_system.beliefs.get('curiosity').mu + curiosity_boost)
            errors['curiosity'] = self.belief_system.update_belief(
                'curiosity', new_curiosity, obs_sigma=0.15
            )
        
        # Confidence affected by quality of interaction
        current_confidence = self.belief_system.beliefs.get('confidence').mu
        if quality_score > 0.3:
            # High quality interactions boost confidence
            new_confidence = min(0.9, current_confidence + 0.05)
        elif quality_score < -0.3:
            # Low quality interactions reduce confidence
            new_confidence = max(0.2, current_confidence - 0.05)
        else:
            new_confidence = current_confidence
        
        errors['confidence'] = self.belief_system.update_belief(
            'confidence', new_confidence, obs_sigma=0.2
        )
        
        # Update user engagement belief with quality awareness
        if 'engagement_signal' in analysis:
            # Weight engagement by quality
            adjusted_engagement = analysis['engagement_signal'] * (0.7 + quality_score * 0.3)
            engagement = BeliefState(mu=adjusted_engagement, sigma=0.2)
            self.user_beliefs['engagement'] = engagement
        
        # Update interaction quality belief
        current_quality_belief = self.user_beliefs['interaction_quality']
        new_quality_mu = (current_quality_belief.mu * 0.7) + ((quality_score + 1.0) / 2.0 * 0.3)
        self.user_beliefs['interaction_quality'] = BeliefState(
            mu=new_quality_mu,
            sigma=max(0.1, current_quality_belief.sigma * 0.95)
        )
        
        # Track topic familiarity
        if 'topic' in analysis:
            self.current_topic = analysis['topic']
            if analysis['topic'] not in self.topic_beliefs:
                # New topic - start uncertain
                self.topic_beliefs[analysis['topic']] = BeliefState(mu=0.3, sigma=0.4)
            else:
                # Increase familiarity with topic (faster if high quality)
                topic_belief = self.topic_beliefs[analysis['topic']]
                familiarity_boost = 0.1 if quality_score > 0 else 0.05
                topic_belief.mu = min(1.0, topic_belief.mu + familiarity_boost)
                topic_belief.sigma = max(0.1, topic_belief.sigma * 0.95)
        
        # Update surprise accumulation
        total_surprise = sum(abs(e) for e in errors.values() if e is not None)
        self.accumulated_surprise += total_surprise
        self.messages_since_reflection += 1
        
        return errors, solvency_adjustment
    
    def deplete_resources_quality_aware(
        self, 
        response_length: int, 
        topic_novelty: float = 0.0,
        response_quality: float = 0.0
    ):
        """
        Deplete resources after generating a response, considering quality.
        
        Args:
            response_length: Length of generated response
            topic_novelty: How novel/unfamiliar the topic is
            response_quality: Quality of the response we generated (-1 to 1)
        """
        # Calculate complexity based on response length and novelty
        base_complexity = min(1.0, response_length / 2000)
        
        # Novel topics require more effort
        if topic_novelty > 0.5:
            base_complexity *= 1.3
        
        # Quality modulates depletion
        # High quality responses are energizing (reduce depletion)
        # Low quality responses are draining (increase depletion)
        quality_modifier = 1.0 - (response_quality * 0.3)
        adjusted_complexity = base_complexity * quality_modifier
        
        # Process the interaction cost
        result = self.belief_system.process_interaction(
            complexity=adjusted_complexity,
            quality=max(0, response_quality)  # Only positive quality gains
        )
        
        # Energy depletes with effort (but less if high quality)
        current_energy = self.belief_system.beliefs.get('energy').mu
        energy_cost = adjusted_complexity * 0.1
        new_energy = max(0.1, current_energy - energy_cost)
        self.belief_system.update_belief('energy', new_energy, obs_sigma=0.1)
        
        # Update interaction tracking
        self.last_interaction_time = time.time()
        self.interaction_count += 1
    
    def recover(self, hours_elapsed: float):
        """
        Enhanced recovery dynamics that consider interaction quality history.
        
        Args:
            hours_elapsed: Time since last interaction in hours
        """
        recovery_rate = min(hours_elapsed / 24.0, 1.0)
        
        # Base recovery modified by average interaction quality
        quality_modifier = 1.0
        if self.quality_history:
            avg_quality = np.mean(self.quality_history)
            # Good quality interactions lead to better recovery
            quality_modifier = 1.0 + (avg_quality * 0.3)
        
        adjusted_recovery = recovery_rate * quality_modifier
        
        # Energy recovers (better with high quality history)
        current_energy = self.belief_system.beliefs.get('energy').mu
        recovered_energy = min(1.0, current_energy + (adjusted_recovery * 0.3))
        self.belief_system.update_belief('energy', recovered_energy, obs_sigma=0.05)
        
        # Social satiation decays
        current_social = self.belief_system.beliefs.get('social_satiation').mu
        decayed_social = max(0.0, current_social - (recovery_rate * 0.2))
        self.belief_system.update_belief('social_satiation', decayed_social, obs_sigma=0.1)
        
        # Confidence recovers if quality was good
        if self.quality_history and np.mean(self.quality_history) > 0.3:
            current_confidence = self.belief_system.beliefs.get('confidence').mu
            recovered_confidence = min(0.8, current_confidence + (adjusted_recovery * 0.1))
            self.belief_system.update_belief('confidence', recovered_confidence, obs_sigma=0.1)
        
        # All beliefs drift toward higher uncertainty
        for name, belief in self.belief_system.beliefs.beliefs.items():
            belief.sigma = min(0.5, belief.sigma * (1.0 + recovery_rate * 0.1))
        
        # Solvency recovers toward target (better with quality history)
        target_solvency = 0.7 + (np.mean(self.quality_history) * 0.2 if self.quality_history else 0)
        current_solvency = self.belief_system.modulator.solvency
        recovered_solvency = current_solvency + (target_solvency - current_solvency) * adjusted_recovery * 0.3
        self.belief_system.modulator.solvency = max(0.2, min(0.9, recovered_solvency))
        
        # Curiosity refreshes
        current_curiosity = self.belief_system.beliefs.get('curiosity').mu
        refreshed_curiosity = current_curiosity + (adjusted_recovery * 0.1)
        self.belief_system.update_belief('curiosity', min(0.8, refreshed_curiosity), obs_sigma=0.1)
        
        # Quality momentum decays during rest
        self.quality_momentum *= (1.0 - recovery_rate * 0.5)
    
    def should_trigger_reflection(self) -> bool:
        """Check if deep reflection (System 2) should trigger."""
        SURPRISE_THRESHOLD = 2.0
        MAX_MESSAGES = 50
        LOW_QUALITY_THRESHOLD = -0.2
        HIGH_QUALITY_THRESHOLD = 0.5
        
        # Trigger reflection on normal conditions
        standard_trigger = (
            self.accumulated_surprise > SURPRISE_THRESHOLD or
            self.messages_since_reflection > MAX_MESSAGES
        )
        
        # Also trigger on sustained low quality
        quality_trigger = False
        if len(self.quality_history) >= 5:
            recent_avg = np.mean(self.quality_history[-5:])
            quality_trigger = recent_avg < LOW_QUALITY_THRESHOLD
        
        # Or on exceptionally high quality (to consolidate gains)
        excellence_trigger = False
        if len(self.quality_history) >= 3:
            recent_avg = np.mean(self.quality_history[-3:])
            excellence_trigger = recent_avg > HIGH_QUALITY_THRESHOLD
        
        return standard_trigger or quality_trigger or excellence_trigger
    
    def reset_reflection_counters(self):
        """Reset counters after reflection."""
        self.accumulated_surprise = 0.0
        self.messages_since_reflection = 0
        # Don't reset quality history - it's valuable context
    
    def get_quality_summary(self) -> Dict:
        """Get summary of interaction quality metrics."""
        if not self.quality_history:
            return {
                'current_quality': 0.5,
                'average_quality': 0.5,
                'quality_trend': 'neutral',
                'recommendation': 'No quality data yet'
            }
        
        current = self.quality_history[-1]
        average = np.mean(self.quality_history)
        
        # Determine trend
        if self.quality_momentum > 0.1:
            trend = 'improving'
        elif self.quality_momentum < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Generate recommendation
        if average < 0:
            recommendation = 'Engagement quality is low. Consider more thoughtful, relevant responses.'
        elif average > 0.5:
            recommendation = 'Excellent engagement quality! Keep up the meaningful interaction.'
        else:
            recommendation = 'Moderate engagement quality. Room for deeper connection.'
        
        return {
            'current_quality': round(current, 3),
            'average_quality': round(average, 3),
            'quality_trend': trend,
            'quality_momentum': round(self.quality_momentum, 3),
            'interactions_tracked': len(self.quality_history),
            'cumulative_quality': round(self.cumulative_quality, 3),
            'recommendation': recommendation
        }
    
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
            'quality_history': self.quality_history,
            'cumulative_quality': self.cumulative_quality,
            'quality_momentum': self.quality_momentum,
            'current_topic': self.current_topic,
            'last_interaction_time': self.last_interaction_time,
            'interaction_count': self.interaction_count,
            'accumulated_surprise': self.accumulated_surprise,
            'messages_since_reflection': self.messages_since_reflection,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QualityAdjustedCognitiveState':
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
        
        # Restore quality tracking
        state.quality_history = data.get('quality_history', [])
        state.cumulative_quality = data.get('cumulative_quality', 0.0)
        state.quality_momentum = data.get('quality_momentum', 0.0)
        
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
        quality_info = self.get_quality_summary()
        
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
            'quality_metrics': quality_info,
            'current_topic': self.current_topic,
            'interaction_count': self.interaction_count,
            'needs_reflection': self.should_trigger_reflection()
        }