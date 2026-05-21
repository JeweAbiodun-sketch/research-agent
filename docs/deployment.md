# Deployment

## Render

This repo is configured as a Python web service on Render.

- Build command: `pip install -r requirements.txt`
- Start command: `python agent_server.py`
- The app listens on `PORT`

## n8n

Point the HTTP Request node to:

```text
POST /research
```

Example body:

```json
{
  "company_name": "Anthropic"
}
```

## Generated Outputs

- PDF reports are written to `.generated_reports/`
- Saved workflow artifacts live in `outputs/`

