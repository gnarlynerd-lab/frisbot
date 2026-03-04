# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Frisbot is a conversational AI companion that maintains uncertain beliefs about its internal states using Bayesian inference. The system uses "Cognitive Context Serialization" to translate internal Bayesian states into natural language that shapes LLM responses via the DeepSeek API.

## Development Commands

### Backend (Python FastAPI)
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
python main.py  # Standard version on port 8000
python main_enhanced.py  # Enhanced version with metacognitive features

# Run tests
python test_cognitive.py  # Test Bayesian belief system (no API key needed)
python test_chat_cli.py  # Interactive CLI test
```

### Frontend (Next.js)
```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev  # or ./node_modules/.bin/next dev

# Build production
npm run build

# Start production server  
npm run start

# Lint
npm run lint
```

### Full Stack
```bash
# Start both backend and frontend
./start.sh  # Starts backend on :8000, frontend on :3000
```

## Architecture Overview

### Bayesian Cognitive System
The core innovation is in `frisbot/cognitive/` and `shared/bayesian/`:
- **BeliefState**: Gaussian distributions (mean, uncertainty) for internal states
- **AdaptiveBeliefSystem**: Manages energy, mood, curiosity, confidence beliefs with solvency modulation
- **Solvency**: Meta-variable (0.0-1.0) that modulates precision across all beliefs - high solvency means confident beliefs, low means uncertain

### Backend Structure
- `frisbot/api/`: REST endpoints for chat interactions
  - `chat.py`: Basic chat endpoint
  - `enhanced_chat.py`: Chat with quality-based solvency updates
  - `metacognitive_chat.py`: Full metacognitive system with self-evaluation
- `frisbot/cognitive/`: Bayesian belief management and serialization
- `frisbot/llm/`: DeepSeek API integration and prompting
- `frisbot/models/`: SQLite database for persistence

### Frontend Structure  
- `frontend/app/`: Next.js app router pages
  - `page.tsx`: Basic chat interface
  - `metacognitive.tsx`: Enhanced interface showing belief states
  - `meta/page.tsx`: Full metacognitive view
- Uses React hooks for state management and axios for API calls

## API Configuration
The system requires a DeepSeek API key:
1. Copy `.env.example` to `.env`
2. Add your key: `DEEPSEEK_API_KEY=your_key_here`
3. Get keys at: https://platform.deepseek.com/

## Key Endpoints
- `POST /api/chat`: Send message, get response with belief updates
- `POST /api/enhanced/chat`: Enhanced chat with quality-based solvency
- `POST /api/metacognitive/chat`: Full metacognitive chat with self-evaluation
- `GET /api/companions/{id}`: Get companion state
- `GET /api/companions/{id}/history`: Get conversation history

## Important Implementation Details
- The cognitive state serializer converts Bayesian beliefs into natural language context that guides LLM behavior
- Solvency acts as a resource constraint - low solvency creates naturally degraded, uncertain responses
- The system maintains persistent state in SQLite, storing both belief distributions and conversation history
- Frontend components poll for belief state updates to show real-time changes during conversation