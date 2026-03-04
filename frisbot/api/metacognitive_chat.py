"""
Metacognitive chat API where DeepSeek's self-evaluations drive Bayesian belief updates.
"""

import uuid
import time
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..cognitive.metacognitive_state import DeepSeekMetacognitiveState, MetacognitiveBeliefSystem
from ..llm.metacognitive_client import MetacognitiveDeepSeekClient
from ..models.database import Database


class MetacognitiveRequest(BaseModel):
    message: str
    companion_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    enable_deep_introspection: bool = False


class MetacognitiveResponse(BaseModel):
    response: str
    companion_id: str
    session_id: str
    cognitive_state: Dict
    metacognitive_state: Dict
    message_analysis: Dict
    metacognitive_assessment: Dict
    cognitive_coherence: float
    solvency: float
    introspection: Optional[Dict] = None


class MetacognitiveCompanionService:
    """
    Service where DeepSeek's metacognitive assessments become 
    Bayesian observations that update beliefs.
    """
    
    def __init__(self):
        self.db = Database()
        self.companions: Dict[str, DeepSeekMetacognitiveState] = {}
    
    def get_llm_client(self, api_key: Optional[str] = None) -> MetacognitiveDeepSeekClient:
        """Get metacognitive DeepSeek client."""
        return MetacognitiveDeepSeekClient(api_key=api_key)
    
    async def get_or_create_companion(
        self, 
        companion_id: Optional[str] = None
    ) -> DeepSeekMetacognitiveState:
        """Get or create metacognitive companion."""
        if not companion_id:
            companion_id = str(uuid.uuid4())
            companion = DeepSeekMetacognitiveState(
                companion_id=companion_id,
                belief_system=None  # Will be initialized by __post_init__
            )
            companion.belief_system = companion.belief_system or MetacognitiveBeliefSystem()
            
            self.companions[companion_id] = companion
            
            # Save to database
            self.db.create_companion(
                companion_id=companion_id,
                name="Metacognitive-Frisbot",
                cognitive_state=companion.get_complete_state()
            )
            
            return companion
        
        if companion_id in self.companions:
            return self.companions[companion_id]
        
        # Load from database
        companion_data = self.db.get_companion(companion_id)
        if not companion_data:
            raise HTTPException(status_code=404, detail="Companion not found")
        
        # Restore state (simplified for demo)
        companion = DeepSeekMetacognitiveState(companion_id=companion_id)
        self.companions[companion_id] = companion
        return companion
    
    async def process_metacognitive_chat(
        self,
        message: str,
        companion_id: Optional[str] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_deep_introspection: bool = False
    ) -> MetacognitiveResponse:
        """
        Process chat with metacognitive Bayesian updates.
        
        Key innovation: DeepSeek's self-assessments become observations
        that update beliefs through Bayesian inference.
        """
        # Get companion
        companion = await self.get_or_create_companion(companion_id)
        
        # Get LLM client
        llm = self.get_llm_client(api_key)
        
        # Get conversation history
        history = self.db.get_conversation_history(
            companion.companion_id, 
            limit=10
        ) if companion.companion_id else []
        
        # Get current belief state for metacognition
        current_beliefs = {
            **{name: belief.mu for name, belief in companion.belief_system.beliefs.beliefs.items()},
            **{name: belief.mu for name, belief in companion.belief_system.metacognitive_beliefs.beliefs.items()}
        }
        
        # === STEP 1: DeepSeek analyzes message WITH metacognition ===
        message_analysis, metacognitive_assessment = await llm.analyze_with_metacognition(
            message=message,
            context=[msg['content'] for msg in history[-3:]] if history else None,
            current_beliefs=current_beliefs
        )
        
        # === STEP 2: Process through Bayesian belief system ===
        # This is where DeepSeek's self-assessments become Bayesian observations
        update_results = companion.process_deepseek_introspection(
            message_analysis=message_analysis,
            metacognitive_assessment=metacognitive_assessment
        )
        
        # === STEP 3: Generate response with metacognitive awareness ===
        # Build conversation context
        messages = []
        if history:
            for msg in history[-5:]:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        messages.append({"role": "user", "content": message})
        
        # Generate with metacognitive state awareness
        response_text, post_response_meta = await llm.generate_with_metacognitive_awareness(
            messages=messages,
            metacognitive_state=companion.belief_system.get_metacognitive_summary()['metacognitive_beliefs'],
            temperature=0.7 if companion.belief_system.solvency > 0.5 else 0.5,
            max_tokens=400
        )
        
        # === STEP 4: Update beliefs with post-response assessment ===
        # DeepSeek's assessment of its own response becomes another observation
        post_meta_updates = companion.belief_system.update_from_deepseek_metacognition(
            post_response_meta
        )
        
        # === STEP 5: Optional deep introspection (System 2) ===
        introspection = None
        if enable_deep_introspection or companion.interaction_count % 20 == 0:
            introspection = await llm.perform_deep_introspection(
                conversation_history=history[-10:] if history else [],
                current_state=companion.get_complete_state()
            )
            
            # Apply recommended adjustments from introspection
            if 'recommended_adjustment' in introspection:
                for belief_name, adjustment in introspection['recommended_adjustment'].items():
                    if belief_name in companion.belief_system.beliefs.beliefs:
                        current = companion.belief_system.beliefs.get(belief_name)
                        new_mu = max(0.0, min(1.0, current.mu + adjustment))
                        companion.belief_system.beliefs.beliefs[belief_name].mu = new_mu
        
        # Save interaction to database
        self.db.save_message(
            companion_id=companion.companion_id,
            session_id=session_id or str(uuid.uuid4()),
            role="user",
            content=message,
            message_analysis={
                **message_analysis,
                'metacognitive': metacognitive_assessment
            }
        )
        
        self.db.save_message(
            companion_id=companion.companion_id,
            session_id=session_id or str(uuid.uuid4()),
            role="assistant",
            content=response_text,
            cognitive_state=companion.get_complete_state()
        )
        
        # Update companion state
        companion.interaction_count += 1
        companion.last_interaction_time = time.time()
        
        # Calculate cognitive coherence
        coherence = companion._calculate_cognitive_coherence()
        
        return MetacognitiveResponse(
            response=response_text,
            companion_id=companion.companion_id,
            session_id=session_id or str(uuid.uuid4()),
            cognitive_state={
                name: {'mu': round(belief.mu, 3), 'sigma': round(belief.sigma, 3)}
                for name, belief in companion.belief_system.beliefs.beliefs.items()
            },
            metacognitive_state=companion.belief_system.get_metacognitive_summary(),
            message_analysis=message_analysis,
            metacognitive_assessment=metacognitive_assessment,
            cognitive_coherence=coherence,
            solvency=companion.belief_system.solvency,
            introspection=introspection
        )


# Create router and service
router = APIRouter(prefix="/api/metacognitive", tags=["metacognitive"])
service = MetacognitiveCompanionService()


@router.post("/chat", response_model=MetacognitiveResponse)
async def metacognitive_chat(request: MetacognitiveRequest):
    """
    Chat endpoint where DeepSeek's metacognitive assessments
    drive Bayesian belief updates.
    """
    return await service.process_metacognitive_chat(
        message=request.message,
        companion_id=request.companion_id,
        session_id=request.session_id,
        api_key=request.api_key,
        enable_deep_introspection=request.enable_deep_introspection
    )


@router.get("/companions/{companion_id}/metacognition")
async def get_metacognitive_state(companion_id: str):
    """Get detailed metacognitive state."""
    companion = await service.get_or_create_companion(companion_id)
    
    return {
        "companion_id": companion_id,
        "metacognitive_summary": companion.belief_system.get_metacognitive_summary(),
        "cognitive_coherence": companion._calculate_cognitive_coherence(),
        "complete_state": companion.get_complete_state()
    }


@router.post("/companions/{companion_id}/introspect")
async def trigger_introspection(companion_id: str, api_key: Optional[str] = None):
    """Trigger deep metacognitive introspection."""
    companion = await service.get_or_create_companion(companion_id)
    llm = service.get_llm_client(api_key)
    
    # Get conversation history
    history = service.db.get_conversation_history(companion_id, limit=20)
    
    # Perform deep introspection
    introspection = await llm.perform_deep_introspection(
        conversation_history=history,
        current_state=companion.get_complete_state()
    )
    
    # Apply insights
    if 'recommended_adjustment' in introspection:
        for belief_name, adjustment in introspection['recommended_adjustment'].items():
            if belief_name in companion.belief_system.beliefs.beliefs:
                current = companion.belief_system.beliefs.get(belief_name)
                new_mu = max(0.0, min(1.0, current.mu + adjustment))
                companion.belief_system.beliefs.beliefs[belief_name].mu = new_mu
    
    return {
        "companion_id": companion_id,
        "introspection": introspection,
        "updated_state": companion.get_complete_state()
    }


@router.get("/companions/{companion_id}/belief-dynamics")
async def get_belief_dynamics(companion_id: str):
    """Visualize how metacognitive and object-level beliefs interact."""
    companion = await service.get_or_create_companion(companion_id)
    
    # Calculate influences
    influences = companion.belief_system.compute_metacognitive_influence_on_object_beliefs()
    
    # Get both belief levels
    object_beliefs = {
        name: {'mu': belief.mu, 'sigma': belief.sigma}
        for name, belief in companion.belief_system.beliefs.beliefs.items()
    }
    
    meta_beliefs = {
        name: {'mu': belief.mu, 'sigma': belief.sigma}
        for name, belief in companion.belief_system.metacognitive_beliefs.beliefs.items()
    }
    
    return {
        "companion_id": companion_id,
        "object_level_beliefs": object_beliefs,
        "meta_level_beliefs": meta_beliefs,
        "metacognitive_influences": influences,
        "cognitive_coherence": companion._calculate_cognitive_coherence(),
        "solvency": companion.belief_system.solvency
    }