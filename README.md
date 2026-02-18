# My Ai Bot

This repository contains a simple AI bot project.


Quick start:

1. Copy the example env file and fill keys: `cp .env.example .env` (Windows: copy `.env.example` .env)
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python bot.py` (or the full venv path on Windows)

Required environment variables (in `.env`):
- `BOT_TOKEN` — Telegram bot token
- `GROQ_API_KEY` — Groq API key for LLM, voice and vision features (optional)
- `HF_API_KEY` — Hugging Face API key (optional, used for image services)

If `GROQ_API_KEY` or `BOT_TOKEN` are missing, the bot will run in a degraded mode and print helpful warnings.

Note: `nova_memory.json` and `nova_brain.json` are ignored by default.
