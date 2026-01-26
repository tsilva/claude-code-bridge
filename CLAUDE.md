# claude-code-bridge

Bridge OpenAI tools to Claude Code SDK. Uses your active Claude subscription.

## Quick Start

```bash
uv pip install -e .
claude-code-bridge
```

## Endpoints

- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Model Selection

Model selection uses your Claude Code user settings. The `model` parameter in requests is logged but not used for routing. Configure your default model in Claude Code settings.

## Architecture

```
claude_code_bridge/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── pool.py           # Client pool for connection reuse
├── models.py         # Pydantic models for OpenAI request/response format
├── client.py         # CLI client
└── session_logger.py # Request/response logging
```

## Key Implementation Details

- **Client Pool**: Pre-spawns `ClaudeSDKClient` instances for reduced latency. Uses `/clear` command between requests to reset conversation state while keeping subprocesses warm.
- **Concurrency**: Pool size controls concurrent requests (default: 3, configurable via `POOL_SIZE` env var)
- **Streaming**: SSE format matching OpenAI's streaming response
- **Model selection**: Uses user's Claude Code settings (per-request model selection not supported with pooling)
- **User settings**: Uses `setting_sources=["user"]` to load user's Claude Code settings (including default model)
- **System prompt**: Uses `system_prompt={"type": "preset", "preset": "claude_code"}` to preserve the default Claude Code system prompt

## Environment Variables

- `POOL_SIZE` - Number of pooled clients (default: 3)
- `CLAUDE_TIMEOUT` - Request timeout in seconds (default: 120)
- `PORT` - Server port (default: 8000)

## Testing

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Usage with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `claude-agent-sdk` - Claude Code SDK for Python

## README.md

Keep README.md updated with any significant project changes.
