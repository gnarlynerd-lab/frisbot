"""
Metacognitive state management where DeepSeek's self-evaluations 
directly participate in Bayesian belief updates.

Core innovation: DeepSeek doesn't just evaluate quality externally,
but provides metacognitive assessments that become observations for 
the belief system itself.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from shared.bayesian import BeliefState, BeliefCollection, AdaptiveBeliefSystem, BayesianUpdater


@dataclass
class MetacognitiveBeliefSystem:
    """
    A belief system that incorporates DeepSeek's metacognitive evaluations
    as observations for Bayesian updates.
    
    Key insight: DeepSeek's self-assessments about its own cognitive state
    become "observations" that update beliefs through Bayesian inference.
    """
    
    def __init__(self, initial_solvency: float = 0.7):
        """Initialize metacognitive belief system."""
        self.beliefs = BeliefCollection()
        self.metacognitive_beliefs = BeliefCollection()  # Beliefs about own cognition
        self.solvency = initial_solvency
        
        # Initialize core beliefs
        self._initialize_beliefs()
        
        # Track metacognitive observations from DeepSeek
        self.metacognitive_history = []
        
    def _initialize_beliefs(self):
        """Initialize both object-level and meta-level beliefs."""
        # Object-level beliefs (about the world/conversation)
        self.beliefs.add('energy', mu=0.7, sigma=0.15)
        self.beliefs.add('mood', mu=0.6, sigma=0.20)
        self.beliefs.add('curiosity', mu=0.5, sigma=0.25)
        self.beliefs.add('confidence', mu=0.5, sigma=0.30)
        self.beliefs.add('social_satiation', mu=0.3, sigma=0.20)
        
        # Meta-level beliefs (about own cognitive state)
        self.metacognitive_beliefs.add('understanding_accuracy', mu=0.5, sigma=0.30)
        self.metacognitive_beliefs.add('response_coherence', mu=0.6, sigma=0.25)
        self.metacognitive_beliefs.add('cognitive_load', mu=0.4, sigma=0.20)
        self.metacognitive_beliefs.add('epistemic_confidence', mu=0.5, sigma=0.35)
        self.metacognitive_beliefs.add('model_uncertainty', mu=0.5, sigma=0.30)
    
    def update_from_deepseek_metacognition(
        self,
        metacognitive_assessment: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Update beliefs using DeepSeek's metacognitive self-assessment.
        
        DeepSeek evaluates its own:
        - Understanding of the conversation
        - Confidence in its responses
        - Cognitive load/effort required
        - Epistemic uncertainty
        
        These become observations that update metacognitive beliefs.
        
        Args:
            metacognitive_assessment: DeepSeek's self-evaluation
            
        Returns:
            Dictionary of belief updates with (prediction_error, new_uncertainty)
        """
        updates = {}
        
        # Update understanding accuracy belief
        if 'understanding_confidence' in metacognitive_assessment:
            obs = metacognitive_assessment['understanding_confidence']
            # DeepSeek's confidence in understanding becomes an observation
            # Higher confidence = lower observation uncertainty
            obs_sigma = 0.15 if obs > 0.7 else 0.25
            
            prior = self.metacognitive_beliefs.get('understanding_accuracy')
            posterior, error = BayesianUpdater.update(prior, obs, obs_sigma)
            
            # CRUCIAL: Solvency modulates how much we trust this update
            solvency_factor = 0.5 + (self.solvency * 1.0)
            adjusted_sigma = posterior.sigma / solvency_factor
            
            self.metacognitive_beliefs.beliefs['understanding_accuracy'] = BeliefState(
                mu=posterior.mu,
                sigma=adjusted_sigma
            )
            updates['understanding_accuracy'] = (error, adjusted_sigma)
        
        # Update response coherence belief
        if 'response_quality_self_assessment' in metacognitive_assessment:
            obs = metacognitive_assessment['response_quality_self_assessment']
            obs_sigma = 0.20
            
            prior = self.metacognitive_beliefs.get('response_coherence')
            posterior, error = self._solvency_modulated_update(prior, obs, obs_sigma)
            self.metacognitive_beliefs.beliefs['response_coherence'] = posterior
            updates['response_coherence'] = (error, posterior.sigma)
        
        # Update cognitive load belief
        if 'processing_difficulty' in metacognitive_assessment:
            obs = metacognitive_assessment['processing_difficulty']
            obs_sigma = 0.18
            
            prior = self.metacognitive_beliefs.get('cognitive_load')
            posterior, error = self._solvency_modulated_update(prior, obs, obs_sigma)
            self.metacognitive_beliefs.beliefs['cognitive_load'] = posterior
            updates['cognitive_load'] = (error, posterior.sigma)
            
            # High cognitive load depletes solvency
            if obs > 0.7:
                self.solvency = max(0.1, self.solvency - 0.05)
        
        # Update epistemic confidence
        if 'epistemic_status' in metacognitive_assessment:
            obs = metacognitive_assessment['epistemic_status']
            obs_sigma = 0.25
            
            prior = self.metacognitive_beliefs.get('epistemic_confidence')
            posterior, error = self._solvency_modulated_update(prior, obs, obs_sigma)
            self.metacognitive_beliefs.beliefs['epistemic_confidence'] = posterior
            updates['epistemic_confidence'] = (error, posterior.sigma)
        
        # Model uncertainty (how uncertain DeepSeek is about its own responses)
        if 'model_uncertainty' in metacognitive_assessment:
            obs = metacognitive_assessment['model_uncertainty']
            obs_sigma = 0.20
            
            prior = self.metacognitive_beliefs.get('model_uncertainty')
            posterior, error = self._solvency_modulated_update(prior, obs, obs_sigma)
            self.metacognitive_beliefs.beliefs['model_uncertainty'] = posterior
            updates['model_uncertainty'] = (error, posterior.sigma)
            
            # High model uncertainty increases all other uncertainties
            if obs > 0.6:
                self._propagate_uncertainty(factor=1.1)
        
        # Store metacognitive observation
        self.metacognitive_history.append({
            'timestamp': time.time(),
            'assessment': metacognitive_assessment,
            'updates': updates,
            'solvency': self.solvency
        })
        
        return updates
    
    def _solvency_modulated_update(
        self, 
        prior: BeliefState, 
        observation: float, 
        obs_sigma: float
    ) -> Tuple[BeliefState, float]:
        """
        Perform Bayesian update with solvency modulation.
        
        Key: When solvency is low, we trust observations less.
        """
        # Solvency affects how much we trust the observation
        solvency_factor = 0.5 + (self.solvency * 1.0)
        
        # Adjust observation uncertainty based on solvency
        adjusted_obs_sigma = obs_sigma / solvency_factor
        
        # Standard Bayesian update with adjusted uncertainty
        posterior, error = BayesianUpdater.update(prior, observation, adjusted_obs_sigma)
        
        # Also modulate the posterior uncertainty
        posterior.sigma = posterior.sigma / np.sqrt(solvency_factor)
        
        return posterior, error
    
    def _propagate_uncertainty(self, factor: float):
        """
        Propagate uncertainty through the belief system.
        
        When model uncertainty is high, all beliefs become less certain.
        """
        for name, belief in self.beliefs.beliefs.items():
            belief.sigma = min(0.5, belief.sigma * factor)
        
        for name, belief in self.metacognitive_beliefs.beliefs.items():
            if name != 'model_uncertainty':  # Don't propagate to itself
                belief.sigma = min(0.5, belief.sigma * factor)
    
    def compute_metacognitive_influence_on_object_beliefs(self) -> Dict[str, float]:
        """
        Compute how metacognitive beliefs influence object-level beliefs.
        
        This is where DeepSeek's self-awareness affects actual behavior.
        
        Returns:
            Influence factors for each object-level belief
        """
        influences = {}
        
        # Understanding accuracy affects confidence
        understanding = self.metacognitive_beliefs.get('understanding_accuracy')
        if understanding:
            confidence_influence = understanding.mu * 0.3  # 30% influence
            current_confidence = self.beliefs.get('confidence')
            if current_confidence:
                new_mu = current_confidence.mu * (1 - 0.3) + confidence_influence
                self.beliefs.beliefs['confidence'] = BeliefState(
                    mu=new_mu,
                    sigma=current_confidence.sigma
                )
                influences['confidence'] = confidence_influence
        
        # Response coherence affects mood
        coherence = self.metacognitive_beliefs.get('response_coherence')
        if coherence:
            mood_influence = coherence.mu * 0.2  # 20% influence
            current_mood = self.beliefs.get('mood')
            if current_mood:
                new_mu = current_mood.mu * 0.8 + mood_influence
                self.beliefs.beliefs['mood'] = BeliefState(
                    mu=new_mu,
                    sigma=current_mood.sigma
                )
                influences['mood'] = mood_influence
        
        # Cognitive load affects energy
        load = self.metacognitive_beliefs.get('cognitive_load')
        if load:
            energy_drain = load.mu * -0.15  # High load drains energy
            current_energy = self.beliefs.get('energy')
            if current_energy:
                new_mu = max(0.1, current_energy.mu + energy_drain)
                self.beliefs.beliefs['energy'] = BeliefState(
                    mu=new_mu,
                    sigma=current_energy.sigma
                )
                influences['energy'] = energy_drain
        
        # Epistemic confidence affects curiosity
        epistemic = self.metacognitive_beliefs.get('epistemic_confidence')
        if epistemic:
            # Low epistemic confidence increases curiosity
            curiosity_boost = (1.0 - epistemic.mu) * 0.1
            current_curiosity = self.beliefs.get('curiosity')
            if current_curiosity:
                new_mu = min(0.9, current_curiosity.mu + curiosity_boost)
                self.beliefs.beliefs['curiosity'] = BeliefState(
                    mu=new_mu,
                    sigma=current_curiosity.sigma
                )
                influences['curiosity'] = curiosity_boost
        
        return influences
    
    def get_metacognitive_summary(self) -> Dict:
        """Get summary of metacognitive state."""
        meta_beliefs = {}
        for name, belief in self.metacognitive_beliefs.beliefs.items():
            meta_beliefs[name] = {
                'mu': round(belief.mu, 3),
                'sigma': round(belief.sigma, 3),
                'precision': round(1.0 / (belief.sigma ** 2), 3)
            }
        
        # Calculate overall metacognitive health
        avg_confidence = np.mean([
            self.metacognitive_beliefs.get('understanding_accuracy').mu,
            self.metacognitive_beliefs.get('response_coherence').mu,
            self.metacognitive_beliefs.get('epistemic_confidence').mu
        ])
        
        avg_uncertainty = np.mean([
            belief.sigma for belief in self.metacognitive_beliefs.beliefs.values()
        ])
        
        return {
            'metacognitive_beliefs': meta_beliefs,
            'overall_metacognitive_confidence': round(avg_confidence, 3),
            'average_metacognitive_uncertainty': round(avg_uncertainty, 3),
            'cognitive_load': round(self.metacognitive_beliefs.get('cognitive_load').mu, 3),
            'model_uncertainty': round(self.metacognitive_beliefs.get('model_uncertainty').mu, 3),
            'solvency': round(self.solvency, 3),
            'recent_assessments': len(self.metacognitive_history)
        }


@dataclass 
class DeepSeekMetacognitiveState(MetacognitiveBeliefSystem):
    """
    Complete cognitive state with DeepSeek metacognitive integration.
    """
    
    companion_id: str
    created_at: float = field(default_factory=time.time)
    
    # Core belief system with metacognition
    belief_system: MetacognitiveBeliefSystem = field(default_factory=MetacognitiveBeliefSystem)
    
    # Topic and user beliefs
    topic_beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    user_beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    
    # Tracking
    current_topic: Optional[str] = None
    last_interaction_time: Optional[float] = None
    interaction_count: int = 0
    
    def process_deepseek_introspection(
        self,
        message_analysis: Dict,
        metacognitive_assessment: Dict
    ) -> Dict:
        """
        Process both message analysis and DeepSeek's introspection.
        
        Args:
            message_analysis: Standard analysis (topic, sentiment, etc.)
            metacognitive_assessment: DeepSeek's self-evaluation
            
        Returns:
            Complete update summary
        """
        # First, update metacognitive beliefs
        meta_updates = self.belief_system.update_from_deepseek_metacognition(
            metacognitive_assessment
        )
        
        # Then compute metacognitive influence on object beliefs
        influences = self.belief_system.compute_metacognitive_influence_on_object_beliefs()
        
        # Standard message-based updates (now influenced by metacognition)
        object_updates = self._update_from_message(message_analysis)
        
        # Calculate total surprise (both object and meta level)
        total_surprise = 0.0
        for error, _ in meta_updates.values():
            if error:
                total_surprise += abs(error)
        for error in object_updates.values():
            if error:
                total_surprise += abs(error)
        
        # Adjust solvency based on total cognitive coherence
        coherence = self._calculate_cognitive_coherence()
        solvency_adjustment = (coherence - 0.5) * 0.1  # ±10% max
        self.belief_system.solvency = max(0.1, min(1.0, 
            self.belief_system.solvency + solvency_adjustment
        ))
        
        return {
            'metacognitive_updates': meta_updates,
            'object_updates': object_updates,
            'influences': influences,
            'total_surprise': total_surprise,
            'cognitive_coherence': coherence,
            'solvency_adjustment': solvency_adjustment,
            'new_solvency': self.belief_system.solvency
        }
    
    def _update_from_message(self, analysis: Dict) -> Dict[str, float]:
        """Standard message-based belief updates."""
        errors = {}
        beliefs = self.belief_system.beliefs
        
        # Social satiation
        social_update = min(0.9, beliefs.get('social_satiation').mu + 0.05)
        posterior, error = BayesianUpdater.update(
            beliefs.get('social_satiation'), social_update, 0.1
        )
        beliefs.beliefs['social_satiation'] = posterior
        errors['social'] = error
        
        # Mood from sentiment
        if 'sentiment' in analysis:
            mood_influence = 0.6 + (analysis['sentiment'] * 0.2)
            posterior, error = BayesianUpdater.update(
                beliefs.get('mood'), mood_influence, 0.2
            )
            beliefs.beliefs['mood'] = posterior
            errors['mood'] = error
        
        # Curiosity from novelty
        if 'novelty' in analysis and analysis['novelty'] > 0.5:
            curiosity_boost = min(0.9, beliefs.get('curiosity').mu + 0.1)
            posterior, error = BayesianUpdater.update(
                beliefs.get('curiosity'), curiosity_boost, 0.15
            )
            beliefs.beliefs['curiosity'] = posterior
            errors['curiosity'] = error
        
        return errors
    
    def _calculate_cognitive_coherence(self) -> float:
        """
        Calculate overall cognitive coherence.
        
        High coherence = metacognitive and object beliefs are aligned.
        Low coherence = contradiction/confusion between levels.
        """
        # Compare metacognitive confidence with object-level confidence
        meta_confidence = self.belief_system.metacognitive_beliefs.get('epistemic_confidence').mu
        object_confidence = self.belief_system.beliefs.get('confidence').mu
        confidence_alignment = 1.0 - abs(meta_confidence - object_confidence)
        
        # Compare cognitive load with energy
        cognitive_load = self.belief_system.metacognitive_beliefs.get('cognitive_load').mu
        energy = self.belief_system.beliefs.get('energy').mu
        # High load should correlate with low energy
        load_energy_alignment = 1.0 - abs((1.0 - cognitive_load) - energy)
        
        # Understanding accuracy vs curiosity
        understanding = self.belief_system.metacognitive_beliefs.get('understanding_accuracy').mu
        curiosity = self.belief_system.beliefs.get('curiosity').mu
        # Low understanding should increase curiosity
        understanding_curiosity_alignment = 1.0 - abs((1.0 - understanding) - curiosity * 0.5)
        
        # Weighted average
        coherence = (
            confidence_alignment * 0.4 +
            load_energy_alignment * 0.3 +
            understanding_curiosity_alignment * 0.3
        )
        
        return max(0.0, min(1.0, coherence))
    
    def get_complete_state(self) -> Dict:
        """Get complete cognitive state including metacognition."""
        return {
            'companion_id': self.companion_id,
            'object_beliefs': {
                name: {'mu': belief.mu, 'sigma': belief.sigma}
                for name, belief in self.belief_system.beliefs.beliefs.items()
            },
            'metacognitive_state': self.belief_system.get_metacognitive_summary(),
            'cognitive_coherence': self._calculate_cognitive_coherence(),
            'solvency': self.belief_system.solvency,
            'interaction_count': self.interaction_count
        }