# Frisbot: Open-Source Conversational AI Companion

An AI companion with genuine inner life, built on Bayesian cognitive architecture.

## What Makes Frisbot Different

Unlike standard chatbots, Frisbot maintains **uncertain beliefs about its own internal states** using Bayesian inference. These beliefs (energy, mood, curiosity, confidence) are maintained as Gaussian distributions and updated through interaction. A "solvency" variable modulates precision across all beliefs, creating natural resource dynamics.

The key innovation: **Cognitive Context Serialization** - internal Bayesian states are translated into natural language context that shapes LLM responses, creating behavior that emerges from principled uncertainty rather than scripted responses.

## Quick Start

### Prerequisites
- Python 3.11+
- DeepSeek API key (get from https://platform.deepseek.com/)

### Setup
```bash
# Clone the repository
git clone https://github.com/gnarlynerd-lab/frisbot.git
cd frisbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### Run

#### Test Cognitive System (No API Key Needed)
```bash
python test_cognitive.py
```

#### Interactive Chat
```bash
python test_chat_cli.py
# Choose option 1 for chat (needs API key)
# Choose option 2 for cognitive test (no API key)
```

#### API Server
```bash
python main.py
# Visit http://localhost:8000
```

## Project Structure

```
frisbot/
├── frisbot/             # Main package
│   ├── cognitive/       # Bayesian belief engine
│   ├── llm/            # DeepSeek LLM integration
│   ├── api/            # Chat API endpoints
│   └── models/         # SQLite database
├── shared/             # Reusable Bayesian components
│   └── bayesian/       # Core belief mechanics
├── main.py             # FastAPI application
├── test_chat_cli.py    # Interactive CLI test
├── test_cognitive.py   # Cognitive system tests
├── requirements.txt    # Python dependencies
└── .env.example        # Environment configuration
```

## Core Concepts

### Bayesian Beliefs
Each internal state is represented as a Gaussian distribution N(μ, σ):
- **Mean (μ)**: Current belief about the state
- **Uncertainty (σ)**: How confident the companion is

### Solvency Modulation
The "solvency" variable (0.0-1.0) modulates precision across all beliefs:
- High solvency → confident, precise beliefs
- Low solvency → uncertain, diffuse beliefs

### Cognitive Context
Internal states are serialized into natural language that shapes LLM behavior:
```
"Your energy is low (belief: 0.3 ± 0.2). Keep responses concise."
"You're feeling uncertain about everything right now. Hedge more."
```

## API Endpoints

- `POST /api/chat` - Send message and get response
- `GET /api/companions/{id}` - Get companion state
- `GET /api/companions/{id}/history` - Get conversation history
- `GET /api/companions` - List all companions

## Development Status

**v0.1 (Current)**
- ✅ Bayesian belief engine
- ✅ Cognitive context serialization
- ✅ DeepSeek LLM integration
- ✅ SQLite persistence
- ✅ CLI interface

**v0.2 (Planned)**
- PyMDP deep reflection (System 2)
- Web frontend
- Voice interface
- Multi-companion support

## Why DeepSeek?

DeepSeek offers excellent performance at a fraction of the cost:
- $0.14 per million input tokens
- $0.28 per million output tokens
- Compare to GPT-4: ~100x cheaper
- Compare to Claude: ~50x cheaper

## Research Context

Frisbot implements the Adaptive Stability Architecture (ASA) pattern, treating resource constraints as constitutive of cognition rather than limitations to overcome. This creates genuinely resource-rational behavior that emerges from first principles.

## Contributing

This is an active research project. Contributions welcome!

## License

MIT

## Acknowledgments

Built on research from:
- Active Inference / Free Energy Principle (Friston et al.)
- Adaptive Stability Architecture
- Amortized Bayesian inference patterns

---

*Frisbot is a research project exploring what AI companions should look like architecturally.*