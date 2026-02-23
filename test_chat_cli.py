#!/usr/bin/env python3
"""
CLI test for Frisbot chat functionality.
Interactive conversation to test the complete system.
"""

import asyncio
import json
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent to path
import sys
sys.path.append(os.path.dirname(__file__))

from frisbot.api.chat import CompanionService
from frisbot.cognitive import CognitiveState, CognitiveContextSerializer


async def chat_loop():
    """
    Interactive chat loop for testing Frisbot.
    """
    print("\n" + "=" * 60)
    print("🤖 FRISBOT CLI TEST")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n⚠️  Warning: DEEPSEEK_API_KEY not set!")
        print("   The system will use fallback responses.")
        print("   Set DEEPSEEK_API_KEY in .env file for full LLM features.\n")
    
    # Initialize service
    service = CompanionService()
    
    # Start conversation
    print("\nStarting new conversation...")
    print("Type 'quit' to exit, 'state' to see cognitive state, 'context' for cognitive context\n")
    
    companion_id = None
    session_id = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'state':
                # Show cognitive state
                if companion_id:
                    companion = await service.get_or_create_companion(companion_id)
                    state = companion.get_summary()
                    print("\n📊 Cognitive State:")
                    print(json.dumps(state, indent=2))
                else:
                    print("No active companion yet. Send a message first.")
                continue
            
            elif user_input.lower() == 'context':
                # Show cognitive context
                if companion_id:
                    companion = await service.get_or_create_companion(companion_id)
                    context = CognitiveContextSerializer.serialize(companion, verbose=True)
                    print("\n📝 Cognitive Context:")
                    print(context)
                else:
                    print("No active companion yet. Send a message first.")
                continue
            
            elif user_input.lower() == 'stats':
                # Show statistics
                if companion_id:
                    stats = service.db.get_companion_stats(companion_id)
                    print("\n📈 Statistics:")
                    print(json.dumps(stats, indent=2))
                else:
                    print("No active companion yet. Send a message first.")
                continue
            
            elif not user_input:
                continue
            
            # Process chat message
            print("\n🤔 Thinking...")
            
            response = await service.process_chat(
                message=user_input,
                companion_id=companion_id,
                session_id=session_id
            )
            
            # Update IDs for subsequent messages
            companion_id = response.companion_id
            session_id = response.session_id
            
            # Display response
            print(f"\n🤖 Frisbot: {response.response}")
            
            # Show brief state info
            state = response.state
            print(f"\n   [Solvency: {state['solvency']:.2f} | "
                  f"Energy: {state['energy']['value']:.2f} | "
                  f"Curiosity: {state['curiosity']['value']:.2f} | "
                  f"Topic: {state.get('current_topic', 'general')}]")
            
            # Show if reflection needed
            if response.needs_reflection:
                print("   ⚡ Deep reflection threshold reached (System 2 would trigger)")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Continuing...")


async def test_cognitive_flow():
    """
    Automated test of cognitive flow without LLM.
    """
    print("\n" + "=" * 60)
    print("🧪 TESTING COGNITIVE FLOW (No LLM)")
    print("=" * 60)
    
    # Create companion directly
    companion = CognitiveState(companion_id="test-001")
    print(f"\n✓ Created companion: {companion.companion_id}")
    
    # Test conversation sequence
    test_messages = [
        ("Hello! How are you today?", {'sentiment': 0.8, 'novelty': 0.2, 'engagement_signal': 0.9}),
        ("Tell me about quantum computing", {'sentiment': 0.5, 'novelty': 0.9, 'engagement_signal': 0.7}),
        ("That's interesting", {'sentiment': 0.6, 'novelty': 0.1, 'engagement_signal': 0.5}),
        ("I'm feeling tired", {'sentiment': -0.2, 'novelty': 0.3, 'engagement_signal': 0.4}),
    ]
    
    for message, analysis in test_messages:
        print(f"\n👤 User: {message}")
        
        # Update state
        companion.update_from_message(analysis)
        companion.deplete_resources(100, analysis.get('novelty', 0.5))
        
        # Get context
        context = CognitiveContextSerializer.serialize(companion)
        
        # Show state
        state = companion.get_summary()
        print(f"   Solvency: {state['solvency']:.2f}")
        print(f"   Energy: {state['energy']['value']:.2f}")
        print(f"   Context preview: {context[:100]}...")
    
    print("\n✓ Cognitive flow test complete!")


def main():
    """
    Main entry point for CLI test.
    """
    print("\nFrisbot CLI Test")
    print("1. Interactive chat (requires DEEPSEEK_API_KEY)")
    print("2. Test cognitive flow (no LLM needed)")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(chat_loop())
    elif choice == "2":
        asyncio.run(test_cognitive_flow())
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()