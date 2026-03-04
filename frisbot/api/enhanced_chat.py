"""
Enhanced chat API with quality-based solvency adjustment.
Integrates DeepSeek's quality evaluation into the cognitive loop.
"""

import uuid
import time
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..cognitive.enhanced_state import QualityAdjustedCognitiveState
from ..cognitive.serializer_v2 import CognitivePersonalitySerializer
from ..llm.enhanced_client import EnhancedDeepSeekClient
from ..llm.prompts import build_conversation_prompt
from ..models.database import Database


# Request/Response models
class EnhancedChatRequest(BaseModel):
    message: str
    companion_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    evaluate_quality: bool = True  # Allow disabling quality evaluation


class EnhancedChatResponse(BaseModel):
    response: str
    companion_id: str
    session_id: str
    state: Dict
    analysis: Dict
    quality_score: float
    solvency_adjustment: float
    cognitive_context: str
    quality_summary: Dict
    needs_reflection: bool


class QualityAwareCompanionService:
    """
    Enhanced service layer with quality-based solvency management.
    DeepSeek evaluates interaction quality to inform resource allocation.
    """
    
    def __init__(self):
        self.db = Database()
        self.llm = None  # Lazy init when API key available
        
        # In-memory cache for active companions
        self.active_companions: Dict[str, QualityAdjustedCognitiveState] = {}
    
    def get_llm_client(self, api_key: Optional[str] = None) -> EnhancedDeepSeekClient:
        """Get or create enhanced LLM client with optional API key override."""
        if api_key:
            # Create a new client with the provided API key
            return EnhancedDeepSeekClient(api_key=api_key)
        if not self.llm:
            self.llm = EnhancedDeepSeekClient()
        return self.llm
    
    async def get_or_create_companion(self, companion_id: Optional[str] = None) -> QualityAdjustedCognitiveState:
        """
        Get existing companion or create new one.
        
        Args:
            companion_id: Optional companion ID to retrieve
            
        Returns:
            QualityAdjustedCognitiveState instance
        """
        # If no ID provided, create new companion
        if not companion_id:
            companion_id = str(uuid.uuid4())
            companion = QualityAdjustedCognitiveState(companion_id=companion_id)
            
            # Save to database
            self.db.create_companion(
                companion_id=companion_id,
                name="Frisbot-Q",  # Q for Quality-aware
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
        companion = QualityAdjustedCognitiveState.from_dict(companion_data['cognitive_state'])
        
        # Cache in memory
        self.active_companions[companion_id] = companion
        return companion
    
    async def process_quality_chat(
        self,
        message: str,
        companion_id: Optional[str] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        evaluate_quality: bool = True
    ) -> EnhancedChatResponse:
        """
        Process a chat message with quality-aware cognitive updates.
        
        Args:
            message: User's message
            companion_id: Optional companion ID
            session_id: Optional session ID
            api_key: Optional API key override
            evaluate_quality: Whether to perform quality evaluation
            
        Returns:
            Enhanced chat response with quality metrics
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
        
        # Get current state for quality evaluation
        pre_state = companion.get_summary()
        
        # Analyze message with quality evaluation
        quality_score = 0.0
        solvency_adjustment = 0.0
        
        try:
            llm_client = self.get_llm_client(api_key)
            
            # Get conversation context
            history = self.db.get_conversation_history(companion.companion_id, limit=10)
            context = [msg['content'] for msg in history[-3:]] if history else []
            
            if evaluate_quality:
                # Enhanced analysis with quality evaluation
                analysis, quality_score = await llm_client.analyze_message_with_quality(
                    message=message,
                    context=context,
                    previous_state=pre_state
                )
            else:
                # Fallback to basic analysis without quality
                from ..llm.client import DeepSeekClient
                basic_client = DeepSeekClient(api_key or self.llm.api_key)
                analysis = await basic_client.analyze_message(message, context)
                quality_score = 0.0
                
        except Exception as e:
            print(f"LLM analysis failed: {e}, using fallback")
            # Fallback analysis
            analysis = {
                'topic': 'general',
                'sentiment': 0.0,
                'novelty': 0.5,
                'engagement_signal': 0.5,
                'quality_assessment': {
                    'overall_quality': 0.5
                }
            }
            quality_score = 0.0
        
        # Update cognitive state with quality awareness
        if evaluate_quality:
            prediction_errors, solvency_adjustment = companion.update_from_quality_analysis(
                analysis, quality_score
            )
        else:
            # Fallback to basic update without quality adjustment
            from ..cognitive.state import CognitiveState
            # Simple update without quality
            prediction_errors = {}
            for key in ['social', 'mood', 'curiosity']:
                if key in analysis:
                    prediction_errors[key] = 0.0
        
        # Serialize cognitive context
        cognitive_context = CognitivePersonalitySerializer.serialize_compressed(companion)
        
        # Save user message to database with quality metrics
        message_metadata = {
            **analysis,
            'quality_score': quality_score,
            'solvency_adjustment': solvency_adjustment
        }
        
        self.db.save_message(
            companion_id=companion.companion_id,
            session_id=session_id,
            role="user",
            content=message,
            message_analysis=message_metadata
        )
        
        # Generate response with DeepSeek
        response_quality = 0.0
        
        try:
            llm_client = self.get_llm_client(api_key)
            
            # Build conversation prompt
            messages = build_conversation_prompt(
                cognitive_context=cognitive_context,
                conversation_history=history,
                user_message=message
            )
            
            # Adjust temperature and length based on solvency and quality
            if companion.solvency < 0.3:
                # Low resources: shorter, more focused
                temperature = 0.4
                max_tokens = 200
            elif companion.solvency > 0.7 and quality_score > 0.3:
                # High resources and good quality: more creative
                temperature = 0.8
                max_tokens = 600
            else:
                # Default
                temperature = 0.6
                max_tokens = 400
            
            # Generate response
            response_text = await llm_client.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Evaluate response quality if enabled
            if evaluate_quality:
                response_quality = await llm_client.evaluate_response_quality(
                    user_message=message,
                    assistant_response=response_text,
                    post_state=companion.get_summary()
                )
            
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            # Quality-aware fallback response
            if companion.solvency < 0.3:
                response_text = "I need a moment to gather my thoughts. Could we slow down a bit?"
            elif quality_score < -0.3:
                response_text = "I'm not sure I'm understanding you fully. Could you help me understand better?"
            elif quality_score > 0.5:
                response_text = "That's a wonderful point! I'm really enjoying this conversation."
            else:
                response_text = "I see what you mean. Let me consider that."
        
        # Deplete resources with quality awareness
        companion.deplete_resources_quality_aware(
            response_length=len(response_text),
            topic_novelty=analysis.get('novelty', 0.5),
            response_quality=response_quality
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
        
        # If reflection triggered, reset counters
        if needs_reflection:
            companion.reset_reflection_counters()
            
            # Log reflection trigger reason
            reflection_reason = "standard"
            if companion.quality_history and len(companion.quality_history) >= 5:
                avg_quality = sum(companion.quality_history[-5:]) / 5
                if avg_quality < -0.2:
                    reflection_reason = "low_quality"
                elif avg_quality > 0.5:
                    reflection_reason = "high_quality"
            
            print(f"Reflection triggered for {companion_id}: {reflection_reason}")
        
        return EnhancedChatResponse(
            response=response_text,
            companion_id=companion.companion_id,
            session_id=session_id,
            state=companion.get_summary(),
            analysis=analysis,
            quality_score=quality_score,
            solvency_adjustment=solvency_adjustment,
            cognitive_context=cognitive_context,
            quality_summary=companion.get_quality_summary(),
            needs_reflection=needs_reflection
        )


# Create router and service
router = APIRouter(prefix="/api/v2", tags=["enhanced-chat"])
service = QualityAwareCompanionService()


@router.post("/chat", response_model=EnhancedChatResponse)
async def quality_chat(request: EnhancedChatRequest):
    """
    Enhanced chat endpoint with quality-based solvency adjustment.
    DeepSeek evaluates interaction quality to modulate cognitive resources.
    """
    return await service.process_quality_chat(
        message=request.message,
        companion_id=request.companion_id,
        session_id=request.session_id,
        api_key=request.api_key,
        evaluate_quality=request.evaluate_quality
    )


@router.get("/companions/{companion_id}/quality")
async def get_quality_metrics(companion_id: str):
    """Get detailed quality metrics for a companion."""
    companion = await service.get_or_create_companion(companion_id)
    
    return {
        "companion_id": companion_id,
        "quality_summary": companion.get_quality_summary(),
        "solvency": companion.solvency,
        "quality_history": companion.quality_history,
        "cumulative_quality": companion.cumulative_quality,
        "quality_momentum": companion.quality_momentum,
        "interaction_count": companion.interaction_count
    }


@router.get("/companions/{companion_id}/state")
async def get_companion_state(companion_id: str):
    """Get complete companion state with quality metrics."""
    companion_data = service.db.get_companion(companion_id)
    if not companion_data:
        raise HTTPException(status_code=404, detail="Companion not found")
    
    companion = await service.get_or_create_companion(companion_id)
    stats = service.db.get_companion_stats(companion_id)
    
    return {
        "companion": companion_data,
        "current_state": companion.get_summary(),
        "quality_metrics": companion.get_quality_summary(),
        "stats": stats
    }


@router.post("/companions/{companion_id}/reflect")
async def trigger_reflection(companion_id: str):
    """Manually trigger deep reflection for a companion."""
    companion = await service.get_or_create_companion(companion_id)
    
    # Force reflection
    companion.reset_reflection_counters()
    
    # Update in database
    service.db.update_companion_state(
        companion_id=companion.companion_id,
        cognitive_state=companion.to_dict()
    )
    
    return {
        "companion_id": companion_id,
        "reflection_triggered": True,
        "quality_summary": companion.get_quality_summary(),
        "current_solvency": companion.solvency
    }


@router.get("/companions")
async def list_companions_with_quality():
    """List all companions with quality metrics."""
    companions = service.db.list_companions()
    
    # Enrich with quality data
    enriched = []
    for comp in companions:
        try:
            state = QualityAdjustedCognitiveState.from_dict(comp['cognitive_state'])
            enriched.append({
                **comp,
                'quality_metrics': state.get_quality_summary(),
                'current_solvency': state.solvency
            })
        except:
            enriched.append(comp)
    
    return {"companions": enriched}