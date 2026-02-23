"""
Precision modulation mechanisms for Bayesian beliefs.
Core ASA (Adaptive Stability Architecture) implementation.

The key insight: resource constraints modulate precision (confidence) across all beliefs.
When resources are low, everything becomes uncertain.
"""

from typing import Dict, Optional
from .belief_state import BeliefState, BeliefCollection


class PrecisionModulator:
    """
    Manages precision modulation based on resource state (solvency).
    This is the core ASA mechanism that links resources to uncertainty.
    """
    
    def __init__(self, base_solvency: float = 0.7):
        """
        Initialize with a base solvency level.
        
        Args:
            base_solvency: Initial resource level (0.0 to 1.0)
        """
        self.solvency = base_solvency
        self.solvency_target = base_solvency  # Homeostatic setpoint
        self.solvency_history = []
        
    def get_modulation_factor(self) -> float:
        """
        Calculate precision modulation factor from current solvency.
        
        Returns:
            Modulation factor (typically 0.5 to 1.5)
            - Low solvency (0.0) → factor = 0.5 (halves precision)
            - High solvency (1.0) → factor = 1.5 (increases precision)
        """
        # Linear scaling from 0.5 to 1.5 based on solvency
        return 0.5 + (self.solvency * 1.0)
    
    def get_effective_belief(self, belief: BeliefState) -> Dict[str, float]:
        """
        Get effective belief with solvency-modulated uncertainty.
        
        Args:
            belief: Base belief state
            
        Returns:
            Dictionary with mu, effective_sigma, and precision
        """
        factor = self.get_modulation_factor()
        effective_sigma = belief.sigma / factor
        
        return {
            'mu': belief.mu,
            'sigma': effective_sigma,
            'base_sigma': belief.sigma,
            'precision': 1.0 / (effective_sigma ** 2),
            'modulation_factor': factor,
            'solvency': self.solvency
        }
    
    def update_solvency(self, delta: float) -> None:
        """
        Update solvency level with bounds checking.
        
        Args:
            delta: Change in solvency (can be positive or negative)
        """
        self.solvency = max(0.0, min(1.0, self.solvency + delta))
        self.solvency_history.append(self.solvency)
    
    def apply_homeostasis(self, rate: float = 0.1) -> None:
        """
        Apply homeostatic pressure toward target solvency.
        
        Args:
            rate: Rate of homeostatic return (0.0 to 1.0)
        """
        delta = rate * (self.solvency_target - self.solvency)
        self.update_solvency(delta)
    
    def resource_cost(self, action_complexity: float) -> float:
        """
        Calculate resource cost for an action.
        
        Args:
            action_complexity: Complexity/cost of the action (0.0 to 1.0)
            
        Returns:
            Actual resource cost
        """
        # Base cost scaled by action complexity
        base_cost = 0.02 + (action_complexity * 0.08)
        
        # When already low on resources, costs are amplified
        if self.solvency < 0.3:
            base_cost *= 1.5
        
        return base_cost
    
    def resource_gain(self, interaction_quality: float) -> float:
        """
        Calculate resource gain from positive interaction.
        
        Args:
            interaction_quality: Quality of interaction (0.0 to 1.0)
            
        Returns:
            Resource gain amount
        """
        # Base gain from interaction
        base_gain = interaction_quality * 0.1
        
        # Diminishing returns when already high
        if self.solvency > 0.8:
            base_gain *= 0.5
        
        return base_gain


class AdaptiveBeliefSystem:
    """
    Combines beliefs with precision modulation for a complete ASA system.
    This integrates belief management with resource-based uncertainty.
    """
    
    def __init__(self, initial_solvency: float = 0.7):
        """Initialize with beliefs and precision modulator."""
        self.beliefs = BeliefCollection()
        self.modulator = PrecisionModulator(initial_solvency)
        self.prediction_errors = []
        
    def add_belief(self, name: str, mu: float, sigma: float) -> None:
        """Add a new belief to the system."""
        self.beliefs.add(name, mu, sigma)
    
    def update_belief(self, name: str, observation: float, 
                     obs_sigma: float = 1.0) -> Optional[float]:
        """
        Update a belief with solvency-modulated precision.
        
        Returns:
            Prediction error if belief exists
        """
        belief = self.beliefs.get(name)
        if not belief:
            return None
        
        # Get effective belief with modulated precision
        effective = self.modulator.get_effective_belief(belief)
        effective_belief = BeliefState(
            mu=effective['mu'], 
            sigma=effective['sigma']
        )
        
        # Standard Bayesian update with effective belief
        from .belief_state import BayesianUpdater
        posterior, error = BayesianUpdater.update(
            effective_belief, observation, obs_sigma
        )
        
        # Store the posterior (but with original base uncertainty)
        # This preserves the base uncertainty while using modulated precision for updates
        factor = self.modulator.get_modulation_factor()
        base_sigma = posterior.sigma * factor
        self.beliefs.beliefs[name] = BeliefState(
            mu=posterior.mu,
            sigma=base_sigma
        )
        
        # Track prediction error
        self.prediction_errors.append({
            'belief': name,
            'error': error,
            'solvency': self.modulator.solvency
        })
        
        return error
    
    def process_interaction(self, complexity: float, quality: float) -> Dict[str, float]:
        """
        Process an interaction's effect on resources.
        
        Args:
            complexity: How complex/costly the interaction is
            quality: How positive/enriching the interaction is
            
        Returns:
            Dictionary with cost, gain, and new solvency
        """
        cost = self.modulator.resource_cost(complexity)
        gain = self.modulator.resource_gain(quality)
        
        net_change = gain - cost
        self.modulator.update_solvency(net_change)
        
        return {
            'cost': cost,
            'gain': gain,
            'net_change': net_change,
            'new_solvency': self.modulator.solvency
        }
    
    def get_system_state(self) -> Dict:
        """Get complete system state for serialization."""
        return {
            'beliefs': self.beliefs.to_dict(),
            'solvency': self.modulator.solvency,
            'solvency_target': self.modulator.solvency_target,
            'modulation_factor': self.modulator.get_modulation_factor(),
            'recent_errors': self.prediction_errors[-10:] if self.prediction_errors else []
        }