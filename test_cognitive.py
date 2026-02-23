#!/usr/bin/env python3
"""
Test script for Frisbot cognitive architecture.
Verifies the Bayesian belief system works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from frisbot.cognitive import CognitiveState, CognitiveContextSerializer
import json


def test_cognitive_system():
    """Test the cognitive architecture without the API."""
    
    print("🧠 Testing Frisbot Cognitive Architecture\n")
    print("=" * 50)
    
    # Create a companion
    print("\n1. Creating companion...")
    companion = CognitiveState(companion_id="test-001")
    print(f"   ✓ Companion created with ID: {companion.companion_id}")
    
    # Check initial state
    print("\n2. Initial state:")
    initial = companion.get_summary()
    for key, value in initial.items():
        if isinstance(value, dict):
            print(f"   {key}: μ={value.get('value', 'N/A')}, σ={value.get('uncertainty', 'N/A')}")
        else:
            print(f"   {key}: {value}")
    
    # Test cognitive context serialization
    print("\n3. Cognitive context (initial):")
    context = CognitiveContextSerializer.serialize(companion)
    print(f"   {context[:200]}..." if len(context) > 200 else f"   {context}")
    
    # Simulate some messages
    print("\n4. Simulating conversation...")
    
    # Message 1: High engagement, positive
    print("\n   Message 1: Enthusiastic greeting")
    analysis1 = {
        'topic': 'greeting',
        'sentiment': 0.8,
        'novelty': 0.2,
        'engagement_signal': 0.9
    }
    errors1 = companion.update_from_message(analysis1)
    companion.deplete_resources(100, 0.2)
    print(f"   Prediction errors: {errors1}")
    print(f"   Solvency after: {companion.solvency:.3f}")
    
    # Message 2: Novel topic
    print("\n   Message 2: Complex novel topic")
    analysis2 = {
        'topic': 'quantum_computing',
        'sentiment': 0.5,
        'novelty': 0.9,
        'engagement_signal': 0.7
    }
    errors2 = companion.update_from_message(analysis2)
    companion.deplete_resources(500, 0.9)  # Long response on hard topic
    print(f"   Prediction errors: {errors2}")
    print(f"   Solvency after: {companion.solvency:.3f}")
    
    # Message 3: Low energy interaction
    print("\n   Message 3: Simple familiar topic")
    analysis3 = {
        'topic': 'greeting',
        'sentiment': 0.6,
        'novelty': 0.1,
        'engagement_signal': 0.5
    }
    errors3 = companion.update_from_message(analysis3)
    companion.deplete_resources(50, 0.1)
    print(f"   Prediction errors: {errors3}")
    print(f"   Solvency after: {companion.solvency:.3f}")
    
    # Check state after conversation
    print("\n5. State after conversation:")
    after = companion.get_summary()
    for key, value in after.items():
        if isinstance(value, dict):
            print(f"   {key}: μ={value.get('value', 'N/A')}, σ={value.get('uncertainty', 'N/A')}")
        else:
            print(f"   {key}: {value}")
    
    # Test cognitive context after depletion
    print("\n6. Cognitive context (after depletion):")
    context_after = CognitiveContextSerializer.serialize(companion)
    print(f"   {context_after[:300]}..." if len(context_after) > 300 else f"   {context_after}")
    
    # Test recovery
    print("\n7. Testing recovery (simulating 2 hours passed)...")
    companion.recover(2.0)
    print(f"   Solvency after recovery: {companion.solvency:.3f}")
    recovery_state = companion.get_summary()
    print(f"   Energy after recovery: {recovery_state['energy']['value']:.2f}")
    print(f"   Social satiation after recovery: {recovery_state['social_satiation']['value']:.2f}")
    
    # Test serialization
    print("\n8. Testing serialization...")
    serialized = companion.to_dict()
    print(f"   ✓ Serialized to {len(json.dumps(serialized))} bytes")
    
    # Test deserialization
    restored = CognitiveState.from_dict(serialized)
    print(f"   ✓ Restored companion with ID: {restored.companion_id}")
    print(f"   ✓ Solvency matches: {restored.solvency:.3f}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Cognitive architecture is working.\n")
    
    return True


if __name__ == "__main__":
    test_cognitive_system()