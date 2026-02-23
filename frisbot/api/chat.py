"""
Chat API endpoint for Frisbot.
Integrates cognitive state, LLM, and persistence.
"""

import uuid
import time
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..cognitive import CognitiveState, CognitiveContextSerializer
from ..llm.client import DeepSeekClient
from ..llm.prompts import build_conversation_prompt
from ..models.database import Database


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    companion_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    companion_id: str
    session_id: str
    state: Dict
    analysis: Dict
    cognitive_context: str
    needs_reflection: bool


class CompanionService:
    """
    Service layer for companion management.
    Handles state loading, saving, and LLM interaction.
    """
    
    def __init__(self):
        self.db = Database()
        self.llm = None  # Lazy init when API key available
        
        # In-memory cache for active companions
        self.active_companions: Dict[str, CognitiveState] = {}
    
    def get_llm_client(self) -> DeepSeekClient:
        """Get or create LLM client."""
        if not self.llm:
            self.llm = DeepSeekClient()
        return self.llm
    
    async def get_or_create_companion(self, companion_id: Optional[str] = None) -> CognitiveState:
        """
        Get existing companion or create new one.
        
        Args:
            companion_id: Optional companion ID to retrieve
            
        Returns:
            CognitiveState instance
        """
        # If no ID provided, create new companion
        if not companion_id:
            companion_id = str(uuid.uuid4())
            companion = CognitiveState(companion_id=companion_id)
            
            # Save to database
            self.db.create_companion(
                companion_id=companion_id,
                name="Frisbot",
                cognitive_state=companion.to_dict()
            )
            
            # Cache in memory
            self.active_companions[companion_id] = companion
            return companion
        
        # Check memory cache first
        if companion_id in self.active_companions:
            return self.active_companions[companion_id]
        
        # Load from database
        companion_data = self.db.get_companion(companion_id)
        if not companion_data:
            raise HTTPException(status_code=404, detail="Companion not found")
        
        # Restore state
        companion = CognitiveState.from_dict(companion_data['cognitive_state'])
        
        # Cache in memory
        self.active_companions[companion_id] = companion
        return companion
    
    async def process_chat(
        self,
        message: str,
        companion_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a chat message through the full pipeline.
        
        Args:
            message: User's message
            companion_id: Optional companion ID
            session_id: Optional session ID
            
        Returns:
            Complete chat response
        """
        # Get or create companion
        companion = await self.get_or_create_companion(companion_id)
        
        # Generate session ID if needed
        if not session_id:
            session_id = str(uuid.uuid4())
            # Create session in database
            self.db.create_session(
                session_id=session_id,
                companion_id=companion.companion_id,
                initial_state=companion.to_dict()
            )
        
        # Check for recovery between messages
        if companion.last_interaction_time:
            elapsed = time.time() - companion.last_interaction_time
            if elapsed > 300:  # 5 minutes
                hours_elapsed = elapsed / 3600
                companion.recover(hours_elapsed)
        
        # Analyze message with DeepSeek
        try:
            llm_client = self.get_llm_client()
            
            # Get conversation context
            history = self.db.get_conversation_history(companion.companion_id, limit=10)
            context = [msg['content'] for msg in history[-3:]] if history else []
            
            # Analyze message
            analysis = await llm_client.analyze_message(message, context)
        except Exception as e:
            print(f"LLM analysis failed: {e}, using fallback")
            # Fallback analysis
            analysis = {
                'topic': 'general',
                'sentiment': 0.0,
                'novelty': 0.5,
                'engagement_signal': 0.5
            }
        
        # Update cognitive state from message
        prediction_errors = companion.update_from_message(analysis)
        
        # Serialize cognitive context
        cognitive_context = CognitiveContextSerializer.serialize(companion, verbose=False)
        
        # Save user message to database
        self.db.save_message(
            companion_id=companion.companion_id,
            session_id=session_id,
            role="user",
            content=message,
            message_analysis=analysis
        )
        
        # Generate response with DeepSeek
        try:
            llm_client = self.get_llm_client()
            
            # Build conversation prompt
            messages = build_conversation_prompt(
                cognitive_context=cognitive_context,
                conversation_history=history,
                user_message=message
            )
            
            # Generate response
            response_text = await llm_client.generate_response(
                messages=messages,
                temperature=0.7 if companion.solvency > 0.5 else 0.5,  # Lower temp when depleted
                max_tokens=300 if companion.solvency < 0.3 else 500  # Shorter when exhausted
            )
            
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            # Fallback response based on state
            if companion.solvency < 0.3:
                response_text = "I'm feeling quite depleted right now. Could we keep this brief?"
            elif companion.belief_system.beliefs.get('curiosity').mu > 0.7:
                response_text = "That's interesting! I'd like to understand more about that."
            else:
                response_text = "I understand. Let me think about that for a moment."
        
        # Deplete resources based on response
        companion.deplete_resources(
            response_length=len(response_text),
            topic_novelty=analysis.get('novelty', 0.5)
        )
        
        # Save assistant response to database
        self.db.save_message(
            companion_id=companion.companion_id,
            session_id=session_id,
            role="assistant",
            content=response_text,
            cognitive_state=companion.to_dict()
        )
        
        # Update companion state in database
        self.db.update_companion_state(
            companion_id=companion.companion_id,
            cognitive_state=companion.to_dict()
        )
        
        # Update topic beliefs in database
        if analysis.get('topic') and analysis['topic'] != 'general':
            topic_belief = companion.topic_beliefs.get(analysis['topic'])
            if topic_belief:
                self.db.update_topic_belief(
                    companion_id=companion.companion_id,
                    topic=analysis['topic'],
                    mu=topic_belief.mu,
                    sigma=topic_belief.sigma
                )
        
        # Check if reflection needed
        needs_reflection = companion.should_trigger_reflection()
        
        return ChatResponse(
            response=response_text,
            companion_id=companion.companion_id,
            session_id=session_id,
            state=companion.get_summary(),
            analysis=analysis,
            cognitive_context=cognitive_context,
            needs_reflection=needs_reflection
        )


# Create router and service
router = APIRouter(prefix="/api", tags=["chat"])
service = CompanionService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Processes message through cognitive architecture and LLM.
    """
    return await service.process_chat(
        message=request.message,
        companion_id=request.companion_id,
        session_id=request.session_id
    )


@router.get("/companions/{companion_id}")
async def get_companion(companion_id: str):
    """Get companion state and statistics."""
    companion_data = service.db.get_companion(companion_id)
    if not companion_data:
        raise HTTPException(status_code=404, detail="Companion not found")
    
    stats = service.db.get_companion_stats(companion_id)
    
    return {
        "companion": companion_data,
        "stats": stats
    }


@router.get("/companions/{companion_id}/history")
async def get_history(companion_id: str, limit: int = 20):
    """Get conversation history."""
    history = service.db.get_conversation_history(companion_id, limit=limit)
    return {"history": history}


@router.get("/companions")
async def list_companions():
    """List all companions."""
    companions = service.db.list_companions()
    return {"companions": companions}