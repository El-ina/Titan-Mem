# Titan Memory

A local-first chatbot with atomic memory extraction and knowledge graph visualization.

## Architecture

The project is organized into modular components:

```
titan_memory/
├── config/                      # Configuration files
│   ├── chat_models.yaml         # Chat model settings
│   ├── extraction_models.yaml   # Extraction model settings
│   ├── embedding_models.yaml    # Embedding model settings
│   └── settings.yaml            # General settings
│
├── chat/                        # Chat component
│   ├── adapters.py              # Backend adapters (Ollama, OpenRouter, OpenAI)
│   └── conversation.py           # Chat logic and message handling
│
├── extraction/                  # Memory extraction layer
│   ├── extractor.py             # Core extraction logic
│   ├── prompts.py               # Prompt templates
│   └── adapters.py             # Backend adapters for extraction
│
├── embedding/                   # Embedding component
│   ├── embedder.py             # Text to vector conversion
│   └── adapters.py             # Backend adapters (Ollama, OpenAI)
│
├── knowledge_graph/             # Graph visualization
│   ├── builder.py              # Graph construction
│   └── similarity.py           # Memory similarity calculations
│
├── storage/                     # Data persistence
│   ├── models.py               # Pydantic data models
│   ├── sessions.py             # Session management
│   └── memories.py             # Memory storage
│
├── api/                         # API layer
│   └── routes.py               # FastAPI endpoints
│
├── ui/                          # User interface
│   └── static/
│       └── index.html           # Chat UI
│
└── main.py                      # Application entry point
```

## Getting Started

1. **Configure your models**

Edit the YAML files in `config/`:
- `chat_models.yaml` - Choose your chat backend (Ollama, OpenRouter, OpenAI)
- `extraction_models.yaml` - Choose your extraction model
- `embedding_models.yaml` - Choose your embedding model
- `settings.yaml` - General settings (paths, thresholds)

2. **Start Ollama** (if using local models)

```bash
ollama serve
```

3. **Run the application**

```bash
python3 main.py
```

The UI will be available at `http://127.0.0.1:8000`

## Features

- **Local-first**: All data stored locally in `out/` directory
- **Modular adapters**: Switch between Ollama, OpenRouter, OpenAI by editing YAML files
- **Atomic memory extraction**: Automatically extracts stable, future-useful facts from conversations
- **Knowledge graph**: Visualizes memory connections based on semantic similarity
- **Session management**: Tracks conversations with persistent session IDs

## API Endpoints

- `GET /` - Chat UI
- `GET /graph` - Knowledge graph visualization
- `GET /api/session` - Create new session
- `GET /api/history` - Get session history
- `GET /api/memories` - Get extracted memories
- `POST /api/chat` - Send chat message

## Configuration

### Switching Models

To switch from Ollama to OpenRouter for chat:

1. Edit `config/chat_models.yaml`:
```yaml
current: openrouter
openrouter:
  enabled: true
  api_key: YOUR_API_KEY_HERE
  model: anthropic/claude-3.5-sonnet
```

2. Restart the application

### Settings

Edit `config/settings.yaml` to adjust:
- Similarity thresholds for graph edges
- Output directory paths
- Server host and port

## Data Flow

1. User sends message via chat UI
2. System routes to chat adapter (Ollama/API)
3. AI responds
4. Background task extracts atomic memories
5. Memories are embedded into vectors
6. Knowledge graph is updated with new connections
7. UI refreshes to show updated graph

## Project Philosophy

This project follows a Titans-inspired architecture:
- **Separate core chat from persistent memory**
- Atomic memories are stable, one-sentence facts
- Memory is intentional and auditable
- Local-first approach ensures privacy
