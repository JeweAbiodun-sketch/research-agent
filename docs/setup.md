# Setup

## Local Development

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Add your API keys to `.env`.
4. Start the server with `python agent_server.py`.

## Required Services

- OpenAI
- Pinecone
- Cohere

## Optional Services

- Notion
- Telegram
- Render

## Health Check

Verify the server is running at:

```text
http://127.0.0.1:8000/health
```

