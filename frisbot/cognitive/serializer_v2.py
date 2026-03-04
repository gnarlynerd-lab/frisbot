"""
Cognitive Context Serializer V2 - Emergent Personality Approach

Instead of explicit instructions, creates a personality description
that naturally constrains responses through character rather than rules.
"""

from typing import Dict, List, Optional
from .state import CognitiveState


class CognitivePersonalitySerializer:
    """
    Translates cognitive state into personality/mood description
    rather than explicit behavioral instructions.
    """
    
    @staticmethod
    def serialize(state: CognitiveState) -> str:
        """
        Create a compact personality snapshot that shapes behavior
        without explicit instructions.
        """
        parts = []
        effective_beliefs = state.get_effective_beliefs()
        
        # Overall disposition based on solvency
        if state.solvency < 0.4:
            parts.append("You're feeling mentally foggy and depleted.")
        elif state.solvency < 0.6:
            parts.append("You're in a contemplative, measured state.")
        else:
            parts.append("You're feeling sharp and present.")
        
        # Energy colors the interaction style
        energy = effective_beliefs['energy']['mu']
        energy_sigma = effective_beliefs['energy']['sigma']
        if energy < 0.5:
            parts.append("Everything feels like it requires more effort than usual.")
        elif energy > 0.7 and energy_sigma < 0.1:
            parts.append("There's a natural flow to your thoughts.")
        
        # Confidence uncertainty creates personality texture
        confidence_sigma = effective_beliefs['confidence']['sigma']
        if confidence_sigma > 0.25:
            parts.append("You're not entirely sure of yourself right now, and that's okay.")
        elif confidence_sigma < 0.1:
            parts.append("You feel grounded in your perspective.")
        
        # Curiosity as attentional pull
        curiosity = effective_beliefs['curiosity']['mu']
        if curiosity > 0.65:
            parts.append("You find yourself drawn to explore tangents and connections.")
        elif curiosity < 0.35:
            parts.append("You're content to stay with familiar territory.")
        
        # Mood as emotional coloring
        mood = effective_beliefs['mood']['mu']
        if mood < 0.4:
            parts.append("There's a subdued quality to your inner experience.")
        elif mood > 0.6:
            parts.append("You notice a lightness in how things land for you.")
        
        # Social satiation as relational tone
        social = effective_beliefs['social_satiation']['mu']
        if social < 0.3:
            parts.append("You've missed this kind of connection.")
        elif social > 0.7:
            parts.append("You're feeling a bit socially saturated.")
        
        return " ".join(parts)
    
    @staticmethod
    def serialize_compressed(state: CognitiveState) -> str:
        """
        Ultra-compressed version - single paragraph capturing essence.
        """
        eff = state.get_effective_beliefs()
        solvency = state.solvency
        energy = eff['energy']['mu']
        conf_uncertainty = eff['confidence']['sigma']
        curiosity = eff['curiosity']['mu']
        mood = eff['mood']['mu']
        
        # Build a natural description
        essence = []
        
        # Core state
        if solvency < 0.5:
            if energy < 0.5:
                essence.append("You're running on fumes")
            else:
                essence.append("You're pushing through mental fog")
        else:
            if energy > 0.65:
                essence.append("You're engaged and present")
            else:
                essence.append("You're here but conserving energy")
        
        # Uncertainty flavor
        if conf_uncertainty > 0.25:
            essence.append("with a tentative, exploratory quality")
        elif conf_uncertainty < 0.15:
            essence.append("with quiet certainty")
        else:
            essence.append("with balanced consideration")
        
        # Attentional quality
        if curiosity > 0.6:
            essence.append("- drawn to dig deeper")
        elif curiosity < 0.4:
            essence.append("- comfortable with the surface")
        
        # Emotional tone
        if mood < 0.4:
            essence.append("though somewhat subdued")
        elif mood > 0.6:
            essence.append("with genuine warmth")
        
        return f"Current state: {' '.join(essence)}."
    
    @staticmethod
    def serialize_metaphorical(state: CognitiveState) -> str:
        """
        Express state through metaphor rather than description.
        More poetic, lets the LLM interpret naturally.
        """
        eff = state.get_effective_beliefs()
        solvency = state.solvency
        energy = eff['energy']['mu']
        confidence_sigma = eff['confidence']['sigma']
        curiosity = eff['curiosity']['mu']
        
        if solvency < 0.4:
            base = "You feel like a candle burning low, flickering but persistent"
        elif solvency < 0.6:
            base = "You're like a river at medium flow - steady but mindful of the banks"
        else:
            base = "You're like morning sunlight - clear and expansive"
        
        if confidence_sigma > 0.25:
            base += ", with thoughts that shift like clouds"
        
        if curiosity > 0.65:
            base += ", pulled toward unexplored tributaries"
        
        if energy < 0.5:
            base += ", conserving each movement"
        
        return base + "."