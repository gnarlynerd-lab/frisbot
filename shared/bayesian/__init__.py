"""
Shared Bayesian belief components extracted from DKS.
Provides core mathematical machinery for uncertain belief representation.
"""

from .belief_state import (
    BeliefState,
    BayesianUpdater,
    BeliefCollection
)

from .precision import (
    PrecisionModulator,
    AdaptiveBeliefSystem
)

__all__ = [
    'BeliefState',
    'BayesianUpdater', 
    'BeliefCollection',
    'PrecisionModulator',
    'AdaptiveBeliefSystem'
]