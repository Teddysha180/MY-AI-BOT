---
title: My Ai Bot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# My Ai Bot

This repository contains a simple AI bot project.

Quick start:

1. Copy the example env file and fill keys: `cp .env.example .env` (Windows: `copy .env.example .env`)
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python bot.py` (or the full venv path on Windows)

Required environment variables (in `.env` or host environment):
- `BOT_TOKEN` — Telegram bot token
- `NVIDIA_API_KEY` — NVIDIA NIM API key (`nvapi-...`) for chat, reasoning, and vision
- `GROQ_KEY` (or `GROQ_API_KEY`) — Groq API key for LLM, Whisper voice, and vision features (optional fallback)
- `HF_API_KEY` — Hugging Face API key (optional, used for image services)

Compatibility note: Either `NVIDIA_API_KEY` or `GROQ_KEY` can power AI features. If both are provided, NVIDIA is preferred by default and Groq acts as fallback.

Note: `artovix_memory.json` and `artovix_brain.json` are ignored by default.

## Deploying to Render

This project is configured with `render.yaml` for Render background workers.

Environment variables to set in Render Dashboard:
- `BOT_TOKEN`
- `NVIDIA_API_KEY`
- `GROQ_API_KEY` (optional)
- `HF_API_KEY` (optional)

## Deploying to Heroku

This project includes a `Procfile` so you can run the bot on Heroku as a worker dyno.

Quick Heroku deploy steps:

1. Install the Heroku CLI and log in:

```bash
heroku login
```

2. From the repository root, create a Heroku app (or use an existing app):

```bash
heroku create my-ai-bot
```

3. Set required config vars (replace values):

```bash
heroku config:set BOT_TOKEN=your_telegram_token
heroku config:set NVIDIA_API_KEY=your_nvidia_key
heroku config:set GROQ_KEY=your_groq_key
heroku config:set HF_API_KEY=your_hf_key
```

4. Push to Heroku and scale the worker dyno:

```bash
git push heroku master
heroku ps:scale worker=1
```
