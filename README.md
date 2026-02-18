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
heroku config:set GROQ_API_KEY=your_groq_key
heroku config:set HF_API_KEY=your_hf_key
```

4. Push to Heroku and scale the worker dyno:

```bash
git push heroku master
heroku ps:scale worker=1
```

Notes:
- Heroku stores environment variables securely in the app settings; do not commit your `.env`.
- If your repo's default branch is `main` use `git push heroku main`.
- Monitor logs with `heroku logs --tail`.

Optional: I can add a GitHub Action to auto-deploy to Heroku on push — tell me if you want that.
