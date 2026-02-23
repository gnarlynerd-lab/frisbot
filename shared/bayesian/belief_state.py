"""
Core Bayesian belief state representation.
Extracted from DKS companion_agent.py for reuse.

A belief is represented as a Gaussian distribution N(mu, sigma).
This allows principled uncertainty representation and Bayesian updates.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json


@dataclass
class BeliefState:
    """
    A single belief represented as a Gaussian distribution.
    
    Attributes:
        mu: Mean (expected value) of the belief
        sigma: Standard deviation (uncertainty) of the belief
    """
    mu: float
    sigma: float
    
    def copy(self) -> 'BeliefState':
        """Create a deep copy of this belief state."""
        return BeliefState(mu=self.mu, sigma=self.sigma)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {'mu': self.mu, 'sigma': self.sigma}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BeliefState':
        """Create from dictionary."""
        return cls(mu=data['mu'], sigma=data['sigma'])
    
    @property
    def precision(self) -> float:
        """Precision is inverse of variance (1/sigma^2)."""
        return 1.0 / (self.sigma ** 2)
    
    @property
    def variance(self) -> float:
        """Variance is sigma squared."""
        return self.sigma ** 2
    
    def __repr__(self) -> str:
        return f"BeliefState(μ={self.mu:.3f}, σ={self.sigma:.3f})"


class BayesianUpdater:
    """
    Handles Bayesian belief updates using precision-weighted averaging.
    This is the core mathematical engine for belief evolution.
    """
    
    @staticmethod
    def update(prior: BeliefState, 
               observation: float, 
               obs_sigma: float = 1.0) -> Tuple[BeliefState, float]:
        """
        Perform Bayesian update on a belief given an observation.
        
        Uses precision-weighted averaging:
        - Higher precision (lower uncertainty) = more weight
        - Returns both posterior belief and prediction error
        
        Args:
            prior: Prior belief state
            observation: Observed value
            obs_sigma: Uncertainty of the observation
            
        Returns:
            Tuple of (posterior belief, prediction error)
        """
        # Calculate precisions (inverse variance)
        prior_precision = prior.precision
        obs_precision = 1.0 / (obs_sigma ** 2)
        
        # Prediction error (surprise)
        prediction_error = observation - prior.mu
        
        # Precision-weighted update
        posterior_precision = prior_precision + obs_precision
        posterior_mu = (
            (prior_precision * prior.mu + obs_precision * observation) 
            / posterior_precision
        )
        posterior_sigma = 1.0 / (posterior_precision ** 0.5)
        
        posterior = BeliefState(mu=posterior_mu, sigma=posterior_sigma)
        
        return posterior, prediction_error
    
    @staticmethod
    def update_with_precision_modulation(
        prior: BeliefState,
        observation: float,
        obs_sigma: float,
        modulation_factor: float
    ) -> Tuple[BeliefState, float]:
        """
        Update with precision modulated by an external factor (e.g., solvency).
        
        This is the key ASA mechanism: resource state affects confidence.
        
        Args:
            prior: Prior belief
            observation: Observed value
            obs_sigma: Observation uncertainty
            modulation_factor: Factor to modulate precision (0.5 to 1.5 typical)
            
        Returns:
            Tuple of (posterior belief, prediction error)
        """
        # Modulate the effective uncertainty
        effective_prior_sigma = prior.sigma / modulation_factor
        effective_prior = BeliefState(mu=prior.mu, sigma=effective_prior_sigma)
        
        # Now do standard update with modulated prior
        return BayesianUpdater.update(effective_prior, observation, obs_sigma)


class BeliefCollection:
    """
    Manages a collection of named beliefs.
    Provides convenience methods for batch operations.
    """
    
    def __init__(self, beliefs: Optional[Dict[str, BeliefState]] = None):
        """Initialize with optional initial beliefs."""
        self.beliefs = beliefs or {}
    
    def add(self, name: str, mu: float, sigma: float) -> None:
        """Add a new belief to the collection."""
        self.beliefs[name] = BeliefState(mu=mu, sigma=sigma)
    
    def get(self, name: str) -> Optional[BeliefState]:
        """Get a belief by name."""
        return self.beliefs.get(name)
    
    def update(self, name: str, observation: float, 
               obs_sigma: float = 1.0) -> Optional[float]:
        """
        Update a belief and return prediction error.
        
        Returns:
            Prediction error if belief exists, None otherwise
        """
        if name not in self.beliefs:
            return None
        
        posterior, error = BayesianUpdater.update(
            self.beliefs[name], observation, obs_sigma
        )
        self.beliefs[name] = posterior
        return error
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert all beliefs to dictionary."""
        return {name: belief.to_dict() for name, belief in self.beliefs.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> 'BeliefCollection':
        """Create from dictionary."""
        beliefs = {
            name: BeliefState.from_dict(belief_data) 
            for name, belief_data in data.items()
        }
        return cls(beliefs)
    
    def __repr__(self) -> str:
        return f"BeliefCollection({list(self.beliefs.keys())})"