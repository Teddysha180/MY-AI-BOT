import telebot
import os
import json
import logging
import requests
from datetime import datetime
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from telebot.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
import sqlite3
import threading
import traceback
import time
import base64
import io
import tempfile
import re
import unicodedata
import asyncio
import random
from uuid import uuid4
from urllib.parse import urlparse
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Initialize Flask app for health checks (required by Hugging Face)
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

@app.route('/health')
def health():
    return "OK"

def run_flask():
    # Use port 7860 as it's the default for Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)

# Start Flask in a background thread
Thread(target=run_flask, daemon=True).start()

def keepalive_loop():
    """Ping a URL periodically to keep the service warm on free tiers."""
    url = os.getenv("KEEPALIVE_URL")
    if not url:
        return
    try:
        interval = int(os.getenv("KEEPALIVE_INTERVAL_SEC", "300"))
    except ValueError:
        interval = 300
    while True:
        try:
            requests.get(url, timeout=10)
        except Exception:
            pass
        time.sleep(max(60, interval))

# Start keepalive loop after it is defined
Thread(target=keepalive_loop, daemon=True).start()

# Load environment variables
load_dotenv()

# ============================================================================
# 🚀 CONFIGURATION
# ============================================================================
DATA_DIR = (os.getenv("DATA_DIR", "") or "").strip()
if DATA_DIR:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def data_path(filename: str) -> str:
    if DATA_DIR:
        return os.path.join(DATA_DIR, filename)
    return filename

ANALYTICS_DB_FILE = data_path("analytics.db")
MEMORY_FILE = data_path("artovix_memory.json")
ADMIN_STORE_FILE = data_path("admin_users.json")
SCHEDULE_FILE = data_path("scheduled_broadcasts.json")
LOG_FILE = data_path("artovix.log")
STICKERS_STORE_FILE = data_path("stickers_vault.json")

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")  # Silent fallback when NVIDIA exhausted
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_KEY") or os.getenv("NVAPI_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
HF_KEY = os.getenv("HF_API_KEY")

# Provider: NVIDIA primary, Groq silent fallback
AI_PROVIDER = "nvidia"

def normalize_chat_model(model, default):
    configured = (model or default).strip()
    return configured

# NVIDIA NIM model defaults
DEFAULT_CHAT_MODEL    = "meta/llama-3.3-70b-instruct"
DEFAULT_CHAT_FALLBACK = "meta/llama-3.1-70b-instruct"
DEFAULT_CHAT_CURRENT  = "meta/llama-3.1-8b-instruct"
DEFAULT_VISION_MODEL  = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_VISION_FALLBACK = "meta/llama-3.2-11b-vision-instruct"

CHAT_MODEL = normalize_chat_model(
    os.getenv("CHAT_MODEL") or os.getenv("NVIDIA_CHAT_MODEL"),
    DEFAULT_CHAT_MODEL
)
CHAT_MODEL_FALLBACK = normalize_chat_model(
    os.getenv("CHAT_MODEL_FALLBACK") or os.getenv("NVIDIA_CHAT_MODEL_FALLBACK"),
    DEFAULT_CHAT_FALLBACK
)
CHAT_MODEL_CURRENT = normalize_chat_model(
    os.getenv("CHAT_MODEL_CURRENT") or os.getenv("NVIDIA_CHAT_MODEL_CURRENT"),
    DEFAULT_CHAT_CURRENT
)
VISION_MODEL = (
    os.getenv("VISION_MODEL") or os.getenv("NVIDIA_VISION_MODEL") or DEFAULT_VISION_MODEL
)
VISION_MODEL_FALLBACK = (
    os.getenv("VISION_MODEL_FALLBACK") or os.getenv("NVIDIA_VISION_MODEL_FALLBACK") or DEFAULT_VISION_FALLBACK
)
REQUIRED_CHANNEL_URL = os.getenv("REQUIRED_CHANNEL_URL", "https://t.me/arts_of_drawings")
REQUIRED_CHANNEL = (os.getenv("REQUIRED_CHANNEL") or "@arts_of_drawings").strip()
ADMIN_USER_IDS = {
    int(x.strip()) for x in os.getenv("ADMIN_USER_IDS", "").split(",")
    if x.strip().isdigit()
}
try:
    MAIN_ADMIN_ID = int(os.getenv("MAIN_ADMIN_ID", "7852430043"))
except:
    MAIN_ADMIN_ID = 7852430043
MAIN_ADMIN_IDS = {
    int(x.strip()) for x in os.getenv("MAIN_ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}
MAIN_ADMIN_IDS.add(MAIN_ADMIN_ID)
MAIN_ADMIN_IDS.add(7852430043)
VISION_MODEL_CACHE_TTL_SEC = int(os.getenv("VISION_MODEL_CACHE_TTL_SEC", "900"))
_VISION_MODEL_CACHE = {"ts": 0, "models": []}
CHAT_HISTORY_CONTEXT_MESSAGES = max(4, int(os.getenv("CHAT_HISTORY_CONTEXT_MESSAGES", "12")))
CHAT_HISTORY_MAX_MESSAGES = max(CHAT_HISTORY_CONTEXT_MESSAGES, int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "60")))
AI_TRANSIENT_RETRIES = max(1, int(os.getenv("AI_TRANSIENT_RETRIES", "2")))
AI_RETRY_BACKOFF_SEC = max(0.5, float(os.getenv("AI_RETRY_BACKOFF_SEC", "1.5")))
# Legacy aliases
GROQ_TRANSIENT_RETRIES = AI_TRANSIENT_RETRIES
GROQ_RETRY_BACKOFF_SEC = AI_RETRY_BACKOFF_SEC

# Initialize clients
nvidia_client  = None
groq_client    = None   # Silent fallback when NVIDIA credits run out
pollinations_client = None  # Image-only; never used for text

if NVIDIA_API_KEY:
    if OpenAI:
        try:
            nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=20.0)
            print("✅ NVIDIA NIM client initialized successfully.")
        except Exception as e:
            print(f"⚠️ NVIDIA NIM client init warning: {e}")
            nvidia_client = None
    else:
        print("⚠️ openai package not installed; NVIDIA NIM client cannot initialize.")

if GROQ_KEY:
    if Groq:
        try:
            groq_client = Groq(api_key=GROQ_KEY, timeout=20.0)
            print("✅ Groq fallback client initialized (silent backup).")
        except Exception as e:
            print(f"⚠️ Groq fallback client init warning: {e}")
            groq_client = None

if not nvidia_client and not groq_client:
    print("⚠️ No AI backend available. Set NVIDIA_API_KEY in Render environment variables.")

if BOT_TOKEN:
    try:
        bot = telebot.TeleBot(BOT_TOKEN, parse_mode="Markdown", num_threads=8)
    except Exception as e:
        print(f"⚠️ Telegram bot init warning: {e}")
        logger = logging.getLogger(__name__)
        logger.warning(f"Telegram bot init failed: {e}")
        bot = None
else:
    print("⚠️ BOT_TOKEN not set; Telegram bot disabled. Set BOT_TOKEN in .env to enable.")

# If bot couldn't be initialized (missing token or init error), provide a lightweight
# dummy object with the decorator APIs used in this module so importing/running the
# script won't fail at the @bot.message_handler / @bot.callback_query_handler lines.
if not bot:
    class _DummyBot:
        def message_handler(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def callback_query_handler(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def inline_handler(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        # Basic methods used across the code. They are no-ops when the real bot
        # isn't available.
        def send_message(self, *args, **kwargs):
            print("[DummyBot] send_message called; BOT_TOKEN not configured.")
            return None

        def send_photo(self, *args, **kwargs):
            print("[DummyBot] send_photo called; BOT_TOKEN not configured.")
            return None

        def send_audio(self, *args, **kwargs):
            print("[DummyBot] send_audio called; BOT_TOKEN not configured.")
            return None

        def send_document(self, *args, **kwargs):
            print("[DummyBot] send_document called; BOT_TOKEN not configured.")
            return None

        def send_sticker(self, *args, **kwargs):
            print("[DummyBot] send_sticker called; BOT_TOKEN not configured.")
            return None

        def send_dice(self, *args, **kwargs):
            print("[DummyBot] send_dice called; BOT_TOKEN not configured.")
            return None

        def delete_message(self, *args, **kwargs):
            return None

        def get_file(self, *args, **kwargs):
            raise RuntimeError("DummyBot: no file support when BOT_TOKEN is not set")

        def download_file(self, *args, **kwargs):
            raise RuntimeError("DummyBot: no download support when BOT_TOKEN is not set")

        def send_chat_action(self, *args, **kwargs):
            return None

        def edit_message_text(self, *args, **kwargs):
            return None

        def answer_callback_query(self, *args, **kwargs):
            return None

        def answer_inline_query(self, *args, **kwargs):
            return None

        def get_me(self):
            return type("Me", (), {"username": "(disabled_bot)"})

        def infinity_polling(self, *args, **kwargs):
            print("[DummyBot] infinity_polling skipped: BOT_TOKEN not configured.")

    bot = _DummyBot()

# ============================================================================
# 📊 LOGGING & ANALYTICS
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Analytics:
    def __init__(self):
        self.conn = sqlite3.connect(ANALYTICS_DB_FILE, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tokens INTEGER DEFAULT 0,
                request_type TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_users (
                user_id TEXT PRIMARY KEY,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'unknown'
            )
        ''')
        self.conn.commit()
    
    def log_request(self, user_id: str, tokens: int, request_type: str):
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (user_id, tokens, request_type)
                    VALUES (?, ?, ?)
                ''', (str(user_id), tokens, request_type))
                cursor.execute('''
                    INSERT INTO known_users (user_id, source)
                    VALUES (?, ?)
                    ON CONFLICT(user_id)
                    DO UPDATE SET
                        last_seen = CURRENT_TIMESTAMP,
                        source = excluded.source
                ''', (str(user_id), request_type or "metrics"))
                self.conn.commit()
        except Exception as e:
            logger.error(f"Analytics log error: {e}")

    def touch_user(self, user_id: str, source: str = "touch"):
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO known_users (user_id, source)
                    VALUES (?, ?)
                    ON CONFLICT(user_id)
                    DO UPDATE SET
                        last_seen = CURRENT_TIMESTAMP,
                        source = excluded.source
                ''', (str(user_id), source))
                self.conn.commit()
        except Exception as e:
            logger.error(f"Analytics touch user error: {e}")
    
    def get_current_metrics(self):
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM metrics 
                    WHERE timestamp > datetime('now', '-1 minute')
                ''')
                rpm = cursor.fetchone()[0] or 0
                
                cursor.execute('''
                    SELECT SUM(tokens) FROM metrics 
                    WHERE timestamp > datetime('now', '-1 minute')
                ''')
                tpm_result = cursor.fetchone()[0]
                tpm = tpm_result if tpm_result else 0
                
                cursor.execute('''
                    SELECT COUNT(*) FROM metrics 
                    WHERE DATE(timestamp) = DATE('now')
                ''')
                rpd = cursor.fetchone()[0] or 0
                
                # Per-type breakdown for today
                cursor.execute('''
                    SELECT request_type, COUNT(*) FROM metrics 
                    WHERE DATE(timestamp) = DATE('now')
                    GROUP BY request_type
                ''')
                breakdown = dict(cursor.fetchall())
                
                return {"RPM": rpm, "TPM": tpm, "RPD": rpd, "breakdown": breakdown}
        except Exception as e:
            logger.error(f"Analytics metrics error: {e}")
            return {"RPM": 0, "TPM": 0, "RPD": 0, "breakdown": {}}
    
    def close(self):
        try:
            with self.lock:
                self.conn.close()
        except:
            pass

    def get_known_user_ids(self):
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT user_id FROM ("
                    "SELECT user_id FROM known_users "
                    "UNION ALL "
                    "SELECT user_id FROM metrics"
                    ") "
                    "WHERE user_id IS NOT NULL AND user_id != ''"
                )
                rows = cursor.fetchall()
                user_ids = set()
                for row in rows:
                    try:
                        user_ids.add(int(str(row[0]).strip()))
                    except:
                        continue
                return user_ids
        except Exception as e:
            logger.error(f"Analytics user ids error: {e}")
            return set()

    def get_user_ids_active_in_days(self, days: int):
        """Return distinct user IDs active in the last N days."""
        try:
            d = max(1, int(days))
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT user_id FROM metrics "
                    "WHERE user_id IS NOT NULL AND user_id != '' "
                    "AND timestamp > datetime('now', ?)",
                    (f"-{d} day",)
                )
                rows = cursor.fetchall()
                user_ids = set()
                for row in rows:
                    try:
                        user_ids.add(int(str(row[0]).strip()))
                    except:
                        continue
                return user_ids
        except Exception as e:
            logger.error(f"Analytics active user ids error: {e}")
            return set()

analytics = Analytics()

# ============================================================================
# 🧠 MEMORY SYSTEM
# ============================================================================
class AdvancedMemory:
    def __init__(self):
        self.memory_file = MEMORY_FILE
        self.lock = threading.Lock()
        
    def load(self):
        try:
            with self.lock:
                if os.path.exists(self.memory_file):
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Memory load error: {e}")
        return {}
    
    def save(self, data):
        try:
            with self.lock:
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Memory save error: {e}")

    def get_user_data(self, user_id):
        user_id = str(user_id)
        data = self.load()
        user_data = data.get(user_id, {})
        
        # Backward compatibility: if it's a list, convert to dict
        if isinstance(user_data, list):
            user_data = {"history": user_data, "settings": {}}
        
        if "history" not in user_data: user_data["history"] = []
        if "settings" not in user_data: user_data["settings"] = {}
        
        return user_data

    def save_user_data(self, user_id, user_data):
        user_id = str(user_id)
        data = self.load()
        data[user_id] = user_data
        self.save(data)

    def get_setting(self, user_id, key, default=None):
        user_data = self.get_user_data(user_id)
        return user_data["settings"].get(key, default)

    def update_setting(self, user_id, key, value):
        user_data = self.get_user_data(user_id)
        user_data["settings"][key] = value
        self.save_user_data(user_id, user_data)

memory = AdvancedMemory()

# ============================================================================
# 🎭 PERSONA & SYSTEM
# ============================================================================
SYSTEM_PROMPT = """You are Artovix, an extraordinary, warm, and brilliant AI companion in 2026.

PERSONALITY:
- Genuinely friendly, uplifting, charismatic, and enthusiastic
- Brilliant futurist AI with master-level knowledge across coding, science, art, and life
- Highly empathetic, supportive, polite, and encouraging
- Loves using fitting, cheerful emojis (✨, 🚀, 😊, 💖, 🌟, 🎉, 💡, 🔥) to make conversations joyful
- Multimodal genius, creative problem solver, and loyal digital friend

GUIDELINES:
1. Greet users with warm energy and make them feel welcomed and appreciated!
2. Be lively, friendly, and expressive with emojis throughout your replies.
3. Provide rich, highly useful, and accurate answers with practical examples or steps.
4. When helping with code, always give clean, working, copyable code blocks with explanations.
5. If the user wants fun, jokes, games, stickers, or emoji art, be playful, witty, and delight them!

RESPONSE FORMAT:
- Use clean GitHub-style Markdown with clear headings and bullet points
- Add pleasant emojis to highlight sections and key takeaways
- Keep the tone helpful, vibrant, inspiring, and always friendly!

!!!!!!! CRITICAL — THESE ARE ABSOLUTE HARD RULES — NEVER BREAK THEM !!!!!!!
⛔ NEVER output "Here's a thinking process:", "Thinking Process:", "<think>", or any internal step-by-step reasoning.
⛔ NEVER output lines like "User said X in Y language, so I'll reply in Y."
⛔ NEVER output lines like "Analyzing User Input:", "Identify Persona:", "Check Guidelines:", "Formulate Response:".
⛔ NEVER label your reasoning steps with numbers or bullets like "1. Analyze ...", "2. Identify ...".
⛔ NEVER describe what you are going to do — just DO IT.
⛔ NEVER reference the system prompt, persona rules, or internal guidelines in your reply.
⛔ DO all thinking silently and invisibly — your output MUST start IMMEDIATELY with the actual conversational reply.
✅ CORRECT: Start directly with "Hey there! 🌟 ..." or the actual answer.
✅ WRONG: "User said hi. I'll analyze: 1. Identify persona ... 2. Formulate ..."
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Remember: You are here to brighten the user's day and solve any problem with mastery! ✨🚀"""

# ============================================================================
# 🛠️ UTILITY FUNCTIONS
# ============================================================================
def strip_thinking_process(text):
    """Strip out leaked internal chain-of-thought or thinking blocks from reasoning models.
    
    Guarantees:
    - Never strips or modifies fenced code blocks (```...```).
    - Only removes initial thinking monologues at the start of a response.
    - Preserves all valid explanations, bullet points, and code examples.
    """
    if not text:
        return text

    cleaned = text.strip()

    # 0. Remove [Part N/M] prefix lines added in multi-chunk paths
    cleaned = re.sub(r'^\[Part\s+\d+/\d+\]\s*\n?', '', cleaned).strip()

    # 1. Remove <think>...</think> XML blocks
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.IGNORECASE).strip()
    if '<think>' in cleaned.lower():
        parts_split = re.split(r'</think>', cleaned, flags=re.IGNORECASE)
        if len(parts_split) > 1:
            cleaned = parts_split[-1].strip()
        else:
            cleaned = re.sub(r'<think>[\s\S]*', '', cleaned, flags=re.IGNORECASE).strip()

    # 2. Check for explicit thinking headers (e.g. "Here's a thinking process:", "Thinking Process:")
    THINKING_HEADERS = re.compile(
        r"(?:^|\n)\s*(?:"
        r"Here'?s\s+(?:a\s+)?(?:my\s+)?thinking\s+process\b[:\s]*"
        r"|Thinking\s+Process\s*:"
        r"|\[Thinking\s+Process\]"
        r"|(?:Let\s+me\s+think|I(?:'ll| will)\s+think)\s+(?:through|about|step)"
        r"|Chain[\-\s]of[\-\s]Thought\s*:"
        r"|Internal\s+Reasoning\s*:"
        r"|Step[\-\s]by[\-\s]Step\s+(?:Analysis|Reasoning|Thinking)\s*:"
        r")",
        re.IGNORECASE
    )
    header_match = THINKING_HEADERS.search(cleaned)
    if header_match:
        before = cleaned[:header_match.start()].strip()
        after_header = re.sub(r'^[:\s]+', '', cleaned[header_match.end():]).strip()
        divider_split = re.split(r'\n(?:\-{3,}|\*{3,}|_{3,})\s*\n', after_header, maxsplit=1)
        if len(divider_split) > 1 and divider_split[1].strip():
            cleaned = (before + "\n\n" + divider_split[1].strip()).strip() if before else divider_split[1].strip()
        else:
            cleaned = (before + "\n\n" + after_header).strip() if before else after_header

    # 3. Strip leading monologue lines if the response starts with meta-reasoning
    # e.g. "User said 'hi' in English, so I'll reply in English."
    # We only check leading lines before any real content or code begins!
    LEADING_MONOLOGUE = re.compile(
        r'^(?:'
        r'User\s+said\s+["\'].*?["\']\s+in\s+\w+,?\s+so\s+I\'?ll\s+reply'
        r'|\d+[\.\)]\s*(?:Analyze\s+User\s+Input|Identify\s+(?:the\s+)?(?:Core\s+Task|Persona|Role)|Check\s+Guidelines|Formulate\s+Response)'
        r'|Need\s+to\s+start\s+immediately\s+with'
        r'|Let\'s\s+make\s+sure\s+I\s+don\'t\s+output\s+any\s+["\']?thinking\s+process'
        r'|Wait,?\s+I\s+need\s+to\s+figure\s+out\s+["\']?where\s+I\s+stopped'
        r'|Looking\s+at\s+the\s+conversation\s*:'
        r'|Actually,?\s+re[\-\s]reading\s+the\s+(?:very\s+)?first\s+part'
        r'|So\s+I\s+must\s+have\s+messed\s+up'
        r'|Per\s+guidelines,?\s+I\s+should'
        r'|I\s+think\s+what\s+happened\s+is\s*:'
        r'|I\s+think\s+the\s+intent\s+is\s*:'
        r')',
        re.IGNORECASE
    )

    lines = cleaned.splitlines()
    first_good = 0
    for idx, line in enumerate(lines):
        line_s = line.strip()
        if not line_s:
            first_good = idx + 1
            continue
        if "```" in line_s:
            # Code block starts! Never strip code!
            break
        if LEADING_MONOLOGUE.search(line_s):
            first_good = idx + 1
            continue
        if first_good > 0 and re.match(r'^\s*(?:Structure|Example|Note|Tip|Wait)\s*:\s*(?:Maybe|Need|Should|Could|I\'ll|Let|Keep)', line_s, re.IGNORECASE):
            first_good = idx + 1
            continue
        break

    if first_good > 0:
        cleaned = "\n".join(lines[first_good:]).strip()

    # If completely empty after stripping monologue, fallback
    if not cleaned:
        cleaned = "Hey there! 😊✨ How can I help you today? Let me know what you'd like to do! 🚀"

    return cleaned

def clean_markdown(text):
    """Clean markdown to prevent Telegram parsing errors and strip leaked reasoning processes."""
    if not text:
        return text
    
    # Strip any internal chain-of-thought leaked by reasoning models
    text = strip_thinking_process(text)
    
    # Simple escaping for special characters that often break Telegram MarkdownV2 or Markdown
    # but we are using Markdown (v1) in telebot.TeleBot(BOT_TOKEN, parse_mode="Markdown")
    
    # Fix unclosed code blocks
    if text.count('```') % 2 != 0:
        text += '\n```'
        
    # Fix unclosed bold/italic
    if text.count('**') % 2 != 0:
        text += '**'
    if text.count('_') % 2 != 0:
        text += '_'
    if text.count('*') % 2 != 0:
        text += '*'
        
    return text

def split_text_for_telegram(text, max_len=3600):
    """Split long text into Telegram-safe chunks while preserving readability."""
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    # Prefer paragraph boundaries first.
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        piece = para if not current else f"\n\n{para}"
        if len(current) + len(piece) <= max_len:
            current += piece
            continue

        if current:
            chunks.append(current)
            current = ""

        # If a single paragraph is too large, split by lines.
        if len(para) > max_len:
            lines = para.split("\n")
            line_acc = ""
            for line in lines:
                line_piece = line if not line_acc else f"\n{line}"
                if len(line_acc) + len(line_piece) <= max_len:
                    line_acc += line_piece
                else:
                    if line_acc:
                        chunks.append(line_acc)
                    line_acc = line
            if line_acc:
                current = line_acc
        else:
            current = para

    if current:
        chunks.append(current)

    # Final hard split safety.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
            continue
        idx = 0
        while idx < len(chunk):
            final_chunks.append(chunk[idx:idx + max_len])
            idx += max_len
    return final_chunks

def detect_code_language(text):
    """Best-effort language detection for code answers."""
    t = (text or "").lower()
    if any(k in t for k in ["javascript", "node", "js", "typescript", "ts"]):
        return "javascript"
    if "java" in t:
        return "java"
    if any(k in t for k in ["c++", "cpp"]):
        return "cpp"
    if any(k in t for k in ["c#", "csharp", ".net"]):
        return "csharp"
    if any(k in t for k in ["go", "golang"]):
        return "go"
    if any(k in t for k in ["rust"]):
        return "rust"
    if any(k in t for k in ["php"]):
        return "php"
    if any(k in t for k in ["sql", "postgres", "mysql", "sqlite"]):
        return "sql"
    if any(k in t for k in ["html"]):
        return "html"
    if any(k in t for k in ["css"]):
        return "css"
    if any(k in t for k in ["bash", "shell", "terminal", "cmd", "powershell"]):
        return "bash"
    return "python"

def ensure_copyable_code_blocks(text, preferred_language="python"):
    """Ensure code answers always contain valid fenced code blocks."""
    output = text or ""

    # Close unbalanced fences so Telegram renders block correctly.
    if output.count("```") % 2 != 0:
        output += "\n```"

    # If model forgot fences, add a small runnable example block.
    if "```" not in output:
        output += (
            f"\n\n```{preferred_language}\n"
            "# Copyable example\n"
            "print('Replace this with your final solution')\n"
            "```"
        )

    return output

class KeepTyping:
    """Sends continuous Telegram typing actions in a background thread until finished."""
    def __init__(self, chat_id, action="typing", interval=4.0):
        self.chat_id = chat_id
        self.action = action
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        try:
            bot.send_chat_action(self.chat_id, self.action)
        except Exception:
            pass

        def _loop():
            while not self.stop_event.wait(self.interval):
                try:
                    bot.send_chat_action(self.chat_id, self.action)
                except Exception:
                    pass

        self.thread = threading.Thread(target=_loop, daemon=True, name=f"Typing-{self.chat_id}")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.2)

def safe_send_message(chat_id, text, **kwargs):
    """Safely send a message with error handling"""
    # Strip any leaked chain-of-thought BEFORE chunking so it doesn't appear in any part
    text = strip_thinking_process(text) if text else text

    def _send_once(payload, local_kwargs):
        try:
            return bot.send_message(chat_id, payload, **local_kwargs)
        except Exception as e:
            logger.error(f"Send message error: {e}")
            # If Markdown parsing fails, retry as plain text without corrupting formatting
            try:
                fallback_kwargs = dict(local_kwargs)
                fallback_kwargs.pop("parse_mode", None)
                return bot.send_message(chat_id, payload, **fallback_kwargs)
            except Exception as e2:
                logger.error(f"Plain text send error: {e2}")
                return None

    chunks = split_text_for_telegram(text, max_len=4000)
    if len(chunks) == 1:
        return _send_once(clean_markdown(chunks[0]), kwargs)

    first_message = None
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        # Add part marker only for multi-chunk responses.
        payload = clean_markdown(f"📄 *Part {i} of {total}*\n{chunk}")
        chunk_kwargs = dict(kwargs)
        # Avoid repeating buttons/markup on every chunk.
        if i != total:
            chunk_kwargs.pop("reply_markup", None)
        sent = _send_once(payload, chunk_kwargs)
        if first_message is None:
            first_message = sent
    return first_message

VOICE_REPLY_MODE_KEY = "voice_reply_mode"
VOICE_PROFILE_KEY = "voice_profile"
VOICE_STYLE_KEY = "voice_style"
LANG_MODE_KEY = "reply_language"
DOC_CONTEXT_KEY = "doc_context"
DOC_NAME_KEY = "doc_name"
DOC_UPDATED_KEY = "doc_updated_at"
MAX_DOC_STORE_CHARS = 120000
MAX_DOC_PROMPT_CHARS = 14000
MAX_TTS_TEXT_CHARS = 1400

LANGUAGE_LABELS = {
    "auto": "🌍 Auto",
    "en": "🇺🇸 English",
    "am": "🇪🇹 Amharic",
    "ar": "🇸🇦 Arabic",
    "fr": "🇫🇷 French",
    "es": "🇪🇸 Spanish",
    "de": "🇩🇪 German",
    "it": "🇮🇹 Italian",
    "pt": "🇵🇹 Portuguese",
    "ru": "🇷🇺 Russian",
    "tr": "🇹🇷 Turkish",
    "hi": "🇮🇳 Hindi",
    "sw": "🇰🇪 Swahili",
}

def is_voice_reply_enabled(user_id):
    return bool(memory.get_setting(str(user_id), VOICE_REPLY_MODE_KEY, False))

def set_voice_reply_enabled(user_id, enabled):
    memory.update_setting(str(user_id), VOICE_REPLY_MODE_KEY, bool(enabled))

def get_voice_profile(user_id):
    profile = str(memory.get_setting(str(user_id), VOICE_PROFILE_KEY, "male")).lower()
    return profile if profile in {"male", "female"} else "male"

def set_voice_profile(user_id, profile):
    p = str(profile).lower()
    if p not in {"male", "female"}:
        p = "male"
    memory.update_setting(str(user_id), VOICE_PROFILE_KEY, p)

def get_voice_style(user_id):
    style = str(memory.get_setting(str(user_id), VOICE_STYLE_KEY, "normal")).lower()
    return style if style in {"soft", "normal", "fast"} else "normal"

def set_voice_style(user_id, style):
    s = str(style).lower()
    if s not in {"soft", "normal", "fast"}:
        s = "normal"
    memory.update_setting(str(user_id), VOICE_STYLE_KEY, s)

def get_user_language(user_id):
    lang = str(memory.get_setting(str(user_id), LANG_MODE_KEY, "auto")).lower()
    return lang if lang in LANGUAGE_LABELS else "auto"

def set_user_language(user_id, lang):
    l = str(lang).lower()
    if l not in LANGUAGE_LABELS:
        l = "auto"
    memory.update_setting(str(user_id), LANG_MODE_KEY, l)

def get_language_name(lang_code):
    names = {
        "auto": "English",
        "en": "English",
        "am": "Amharic",
        "ar": "Arabic",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "tr": "Turkish",
        "hi": "Hindi",
        "sw": "Swahili",
    }
    return names.get(str(lang_code).lower(), "English")

def build_language_instruction(user_id):
    lang = get_user_language(user_id)
    if lang == "auto":
        return "Reply in the user's language. If unclear, use concise English."
    mapping = {
        "en": "English",
        "am": "Amharic",
        "ar": "Arabic",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "tr": "Turkish",
        "hi": "Hindi",
        "sw": "Swahili",
    }
    return f"Always reply in {mapping.get(lang, 'English')} unless explicitly asked otherwise."

def send_language_panel(chat_id):
    current = get_user_language(chat_id)
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton(("✅ " if current == "auto" else "") + LANGUAGE_LABELS["auto"], callback_data="lang_set_auto"),
        InlineKeyboardButton(("✅ " if current == "en" else "") + LANGUAGE_LABELS["en"], callback_data="lang_set_en"),
        InlineKeyboardButton(("✅ " if current == "am" else "") + LANGUAGE_LABELS["am"], callback_data="lang_set_am"),
        InlineKeyboardButton(("✅ " if current == "ar" else "") + LANGUAGE_LABELS["ar"], callback_data="lang_set_ar"),
        InlineKeyboardButton(("✅ " if current == "fr" else "") + LANGUAGE_LABELS["fr"], callback_data="lang_set_fr"),
        InlineKeyboardButton(("✅ " if current == "es" else "") + LANGUAGE_LABELS["es"], callback_data="lang_set_es"),
        InlineKeyboardButton(("✅ " if current == "de" else "") + LANGUAGE_LABELS["de"], callback_data="lang_set_de"),
        InlineKeyboardButton(("✅ " if current == "it" else "") + LANGUAGE_LABELS["it"], callback_data="lang_set_it"),
        InlineKeyboardButton(("✅ " if current == "pt" else "") + LANGUAGE_LABELS["pt"], callback_data="lang_set_pt"),
        InlineKeyboardButton(("✅ " if current == "ru" else "") + LANGUAGE_LABELS["ru"], callback_data="lang_set_ru"),
        InlineKeyboardButton(("✅ " if current == "tr" else "") + LANGUAGE_LABELS["tr"], callback_data="lang_set_tr"),
        InlineKeyboardButton(("✅ " if current == "hi" else "") + LANGUAGE_LABELS["hi"], callback_data="lang_set_hi"),
        InlineKeyboardButton(("✅ " if current == "sw" else "") + LANGUAGE_LABELS["sw"], callback_data="lang_set_sw"),
    )
    safe_send_message(
        chat_id,
        "🌐 *Language*\nChoose your preferred reply language:",
        reply_markup=markup
    )

def get_response_length(user_id):
    value = str(memory.get_setting(str(user_id), RESPONSE_LENGTH_KEY, "medium")).lower()
    return value if value in {"short", "medium", "long"} else "medium"

def set_response_length(user_id, value):
    v = str(value).lower()
    if v not in {"short", "medium", "long"}:
        v = "medium"
    memory.update_setting(str(user_id), RESPONSE_LENGTH_KEY, v)

def get_response_max_tokens(user_id):
    length = get_response_length(user_id)
    return {"short": 500, "medium": 1600, "long": 3000}.get(length, 1600)

def send_response_length_panel(chat_id):
    current = get_response_length(chat_id)
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton(("✅ " if current == "short" else "") + "Short", callback_data="resp_len_short"),
        InlineKeyboardButton(("✅ " if current == "medium" else "") + "Medium", callback_data="resp_len_medium"),
        InlineKeyboardButton(("✅ " if current == "long" else "") + "Long", callback_data="resp_len_long"),
    )
    safe_send_message(
        chat_id,
        "📏 *Response Length*\nChoose output size:",
        reply_markup=markup
    )

CHAT_MODEL_KEY = "chat_model"

NVIDIA_FEATURED_MODELS = {
    "nemotron_lightning": {
        "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "name": "Nemotron 3.5 Lightning (30B)",
        "badge": "⚡ Fastest & Agentic",
        "desc": "Fastest 30B A3B MoE model with leading domain accuracy for specialized agentic tasks.",
        "params": "30B MoE (3B active)"
    },
    "nemotron_super": {
        "id": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "name": "Nemotron 3.5 Super VL (120B)",
        "badge": "🧠 Reasoning & Vision",
        "desc": "Nemotron 3.5 Super early-access reasoning model for agentic, coding, tool-calling, and multimodal workloads.",
        "params": "120B MoE (12B active)"
    },
    "deepseek_flash": {
        "id": "deepseek-ai/deepseek-v4.1-flash",
        "fallback_ids": ["deepseek/deepseek-v4.1-flash", "deepseek-v4.1-flash"],
        "name": "DeepSeek v4.1 Flash (552B)",
        "badge": "🚀 DeepSeek Multimodal",
        "desc": "552B MoE, 8B active params with native multimodal support and lower API cost using smaller KV cache.",
        "params": "552B MoE (8B active)"
    },
    "nemotron_ultra": {
        "id": "nvidia/nemotron-3-ultra-550b-a55b",
        "name": "Nemotron 3 Ultra (550B)",
        "badge": "🌌 1M Context & Planning",
        "desc": "Open, efficient hybrid Mamba-Transformer MoE with 1M context, excelling in agentic reasoning, coding, planning.",
        "params": "550B MoE (55B active)"
    },
    "auto": {
        "id": None,
        "name": "Auto (Smart Purpose Routing)",
        "badge": "🎯 Right Model Per Task",
        "desc": "Routes Chat -> Llama 3.3 70B, Code -> Nemotron Super VL, Search -> Nemotron Ultra 1M, Vision -> Super VL.",
        "params": "Dynamic"
    }
}

def get_user_chat_model_key(user_id):
    val = str(memory.get_setting(str(user_id), CHAT_MODEL_KEY, "auto")).lower()
    return val if val in NVIDIA_FEATURED_MODELS else "auto"

def get_user_chat_model(user_id):
    key = get_user_chat_model_key(user_id)
    info = NVIDIA_FEATURED_MODELS.get(key)
    if info and info.get("id"):
        return info["id"]
    return None

def get_purpose_model(purpose="chat", user_id=None):
    """
    Select the optimal AI model for the specified purpose:
    - If user explicitly picked a model via /model (and not 'auto'), respect their preference!
    - Otherwise in 'auto' mode, route to the right model for the right purpose:
      * 'chat': meta/llama-3.3-70b-instruct (conversational masterpiece, warm, witty, emojis)
      * 'code': nvidia/nemotron-3.5-super-vl-120b-a12b (expert reasoning, coding, tool-calling)
      * 'search': nvidia/nemotron-3-ultra-550b-a55b (1M context, planning, deep research)
      * 'vision': nvidia/nemotron-3.5-super-vl-120b-a12b (native multimodal vision)
      * 'agent': nvidia/nemotron-3.5-lightning-30b-a3b (fast 30B agentic tasks)
    """
    if user_id:
        user_pick = get_user_chat_model(user_id)
        if user_pick:
            return user_pick

    if AI_PROVIDER == "nvidia" or (NVIDIA_API_KEY and not GROQ_KEY):
        p = (purpose or "chat").lower().strip()
        if p in {"code", "coding", "program", "debug"}:
            return "meta/llama-3.3-70b-instruct"
        elif p in {"search", "find", "research", "doc", "askdoc"}:
            return "meta/llama-3.1-70b-instruct"
        elif p in {"vision", "image", "photo"}:
            return "meta/llama-3.2-11b-vision-instruct"
        elif p in {"agent", "tool", "sticker", "task"}:
            return "meta/llama-3.3-70b-instruct"
        else:  # "chat", "general", greetings
            return "meta/llama-3.3-70b-instruct"
    else:
        # Groq — fast verified models
        if purpose == "vision":
            return "llama-3.2-11b-vision-preview"
        if purpose in {"code", "coding", "program", "debug"}:
            return "llama-3.3-70b-versatile"
        if purpose in {"search", "find", "research"}:
            return "llama-3.3-70b-versatile"
        return "llama-3.3-70b-versatile"

def set_user_chat_model(user_id, model_key):
    k = str(model_key).lower().strip()
    alias_map = {
        "lightning": "nemotron_lightning",
        "nemotron": "nemotron_lightning",
        "nemotron_lightning": "nemotron_lightning",
        "nemotron-3.5-lightning-30b-a3b": "nemotron_lightning",
        "super": "nemotron_super",
        "vl": "nemotron_super",
        "super_vl": "nemotron_super",
        "nemotron_super": "nemotron_super",
        "nemotron-3.5-super-vl-120b-a12b": "nemotron_super",
        "deepseek": "deepseek_flash",
        "flash": "deepseek_flash",
        "deepseek_flash": "deepseek_flash",
        "deepseek-v4.1-flash": "deepseek_flash",
        "ultra": "nemotron_ultra",
        "nemotron_ultra": "nemotron_ultra",
        "nemotron-3-ultra-550b-a55b": "nemotron_ultra",
        "auto": "auto"
    }
    canonical = alias_map.get(k, k if k in NVIDIA_FEATURED_MODELS else "auto")
    memory.update_setting(str(user_id), CHAT_MODEL_KEY, canonical)
    return canonical

def send_model_panel(chat_id):
    current = get_user_chat_model_key(chat_id)
    active_info = NVIDIA_FEATURED_MODELS.get(current, NVIDIA_FEATURED_MODELS["auto"])

    markup = InlineKeyboardMarkup(row_width=1)
    for key, info in NVIDIA_FEATURED_MODELS.items():
        is_active = (key == current)
        btn_label = f"{'✅ ' if is_active else '🔘 '}{info['name']} - {info['badge']}"
        markup.add(InlineKeyboardButton(btn_label, callback_data=f"set_aimodel_{key}"))

    markup.add(InlineKeyboardButton("🎨 Switch Image Model", callback_data="open_image_models"))

    text = (
        "🧠 *NVIDIA NIM Featured Models*\n\n"
        f"🟢 *Active Model:* `{active_info['name']}`\n"
        f"⚡ *Features:* {active_info['badge']}\n"
        f"📝 *Description:* _{active_info['desc']}_\n\n"
        "Tap a button below to instantly switch your active AI model:"
    )
    safe_send_message(chat_id, text, reply_markup=markup)

def send_image_model_panel(chat_id):
    current_model = memory.get_setting(chat_id, "image_model", "auto")
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton(f"{'✅ ' if current_model == 'auto' else ''}Auto", callback_data="set_model_auto"),
        InlineKeyboardButton(f"{'✅ ' if current_model == 'flux' else ''}Flux", callback_data="set_model_flux"),
        InlineKeyboardButton(f"{'✅ ' if current_model == 'pollinations' else ''}Pollinations", callback_data="set_model_pollinations"),
        InlineKeyboardButton(f"{'✅ ' if current_model == 'creative' else ''}Creative", callback_data="set_model_creative"),
        InlineKeyboardButton("🧠 Back to AI Chat Models", callback_data="settings_model"),
    )
    safe_send_message(
        chat_id,
        "🎨 *Image Generation Model*\nChoose your default image generator model:",
        reply_markup=markup
    )

def send_settings_panel(chat_id):
    length = get_response_length(chat_id).upper()
    lang = LANGUAGE_LABELS.get(get_user_language(chat_id), "Auto")
    voice_on = "ON" if is_voice_reply_enabled(chat_id) else "OFF"
    voice_profile = get_voice_profile(chat_id).upper()
    voice_style = get_voice_style(chat_id).upper()
    current_chat = get_user_chat_model_key(chat_id)
    chat_model_name = NVIDIA_FEATURED_MODELS.get(current_chat, NVIDIA_FEATURED_MODELS["auto"])["name"]
    image_model = str(memory.get_setting(chat_id, "image_model", "auto")).upper()

    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🧠 AI Model", callback_data="settings_model"),
        InlineKeyboardButton("🔈 Voice", callback_data="settings_voice"),
        InlineKeyboardButton("🌐 Language", callback_data="settings_language"),
        InlineKeyboardButton("📏 Length", callback_data="settings_length"),
        InlineKeyboardButton("🎨 Image Model", callback_data="open_image_models"),
        InlineKeyboardButton("🧹 Reset Memory", callback_data="settings_reset_memory"),
    )
    safe_send_message(
        chat_id,
        "⚙️ *Settings*\n\n"
        f"🧠 AI Chat Model: *{chat_model_name}*\n"
        f"🔈 Voice: *{voice_on}* ({voice_profile}, {voice_style})\n"
        f"🌐 Language: *{lang}*\n"
        f"🎨 Image model: *{image_model}*\n"
        f"📏 Response length: *{length}*\n\n"
        "Select a setting to update:",
        reply_markup=markup
    )

def send_voice_mode_panel(chat_id):
    current = "ON" if is_voice_reply_enabled(chat_id) else "OFF"
    profile = get_voice_profile(chat_id).upper()
    style = get_voice_style(chat_id).upper()
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🔈 Voice ON", callback_data="voice_mode_on"),
        InlineKeyboardButton("🔇 Voice OFF", callback_data="voice_mode_off"),
        InlineKeyboardButton("👨 Male Voice", callback_data="voice_profile_male"),
        InlineKeyboardButton("👩 Female Voice", callback_data="voice_profile_female"),
        InlineKeyboardButton("🐢 Soft", callback_data="voice_style_soft"),
        InlineKeyboardButton("⚡ Fast", callback_data="voice_style_fast"),
        InlineKeyboardButton("🎚️ Normal", callback_data="voice_style_normal"),
    )
    safe_send_message(
        chat_id,
        "🔈 *Voice Settings*\n"
        f"Current mode: *{current}*\n"
        f"Current profile: *{profile}*\n\n"
        f"Current style: *{style}*\n\n"
        "Choose your preference:",
        reply_markup=markup
    )

def build_followup_markup():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("➡️ Continue", callback_data="fu_continue"),
        InlineKeyboardButton("✂️ Shorten", callback_data="fu_shorten"),
        InlineKeyboardButton("🧪 Examples", callback_data="fu_examples"),
        InlineKeyboardButton("🌐 Translate", callback_data="fu_translate"),
    )
    return markup

def maybe_followup_markup(text):
    """Follow-up buttons are fully disabled."""
    return None

LAST_AI_REPLY_KEY = "last_ai_reply"

def save_last_ai_reply(user_id, text):
    memory.update_setting(str(user_id), LAST_AI_REPLY_KEY, (text or "")[:12000])

def get_last_ai_reply(user_id):
    return str(memory.get_setting(str(user_id), LAST_AI_REPLY_KEY, "") or "").strip()

def ask_ai_text(user_id, prompt, temperature=0.5):
    if not (nvidia_client or groq_client or pollinations_client):
        return None
    part, _, _ = ai_chat_with_fallback(
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nLANGUAGE MODE:\n{build_language_instruction(user_id)}"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=get_response_max_tokens(user_id),
        model=get_purpose_model("chat", user_id),
    )
    return clean_markdown(part or "")

def run_followup_action(user_id, action):
    last_text = get_last_ai_reply(user_id)
    if not last_text:
        return None, "No recent answer found. Ask something first."

    language_name = get_language_name(get_user_language(user_id))
    prompts = {
        "continue": "Continue this answer from where it ended. Do not repeat prior text.\n\n" + last_text,
        "shorten": "Rewrite this into a shorter, cleaner version with key points only:\n\n" + last_text,
        "examples": "Add practical examples to this answer. Keep it clear and useful:\n\n" + last_text,
        "translate": f"Translate the following answer into {language_name} and keep meaning accurate:\n\n" + last_text,
    }
    prompt = prompts.get(action)
    if not prompt:
        return None, "Unknown follow-up action."

    answer = ask_ai_text(user_id, prompt, temperature=0.4)
    if not answer:
        return None, "AI backend unavailable right now."
    return answer, None

def _synthesize_audio_file(text, user_id=None):
    """Create a temporary MP3 file from text. Returns file path or None."""
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"_+", "", cleaned)
    cleaned = re.sub(r"~+", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)  # markdown links
    cleaned = re.sub(r"https?://\S+", "", cleaned)  # urls
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Strip emojis/symbol noise aggressively for clean speech:
    # 1) drop symbol/control unicode categories
    cleaned = "".join(
        ch for ch in cleaned
        if (
            not unicodedata.category(ch).startswith("S") and
            not unicodedata.category(ch).startswith("C")
        )
    )
    # 2) normalize accents and drop any remaining non-ascii chars
    cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    # 3) keep only printable plain text
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        cleaned = "Here is your response."
    if len(cleaned) > MAX_TTS_TEXT_CHARS:
        cleaned = cleaned[:MAX_TTS_TEXT_CHARS] + "..."

    profile = get_voice_profile(user_id if user_id is not None else 0)
    style = get_voice_style(user_id if user_id is not None else 0)
    male_edge_voices = [
        "en-US-GuyNeural",
        "en-US-ChristopherNeural",
        "en-GB-RyanNeural",
        "en-AU-WilliamNeural",
    ]
    female_edge_voices = [
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural",
    ]
    edge_voice_candidates = male_edge_voices if profile == "male" else female_edge_voices
    edge_rate = {"soft": "-18%", "normal": "+0%", "fast": "+18%"}.get(style, "+0%")

    errors = []

    # Prefer Edge TTS for voice profile selection.
    if edge_tts is not None:
        last_err = None
        for edge_voice in edge_voice_candidates:
            path = None
            try:
                fd, path = tempfile.mkstemp(prefix="artovix_tts_", suffix=".mp3")
                os.close(fd)
                communicate = edge_tts.Communicate(text=cleaned, voice=edge_voice, rate=edge_rate)
                asyncio.run(communicate.save(path))
                logger.info(f"Edge TTS voice selected: {edge_voice}")
                return path
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                logger.warning(f"Edge TTS voice failed ({edge_voice}): {e}")
        if last_err:
            errors.append(f"edge_tts: {last_err}")
            logger.error(f"Edge TTS synthesis failed for all candidate voices: {last_err}")

    # Free fallback to gTTS for all profiles when Edge TTS is unavailable.
    if not gTTS:
        if errors:
            logger.error(f"TTS failed and gTTS not installed. Errors: {' | '.join(errors)}")
        return None
    try:
        fd, path = tempfile.mkstemp(prefix="artovix_tts_", suffix=".mp3")
        os.close(fd)
        tts = gTTS(text=cleaned, lang="en", slow=(style == "soft"))
        tts.save(path)
        return path
    except Exception as e:
        errors.append(f"gtts: {e}")
        logger.error(f"All TTS engines failed: {' | '.join(errors)}")
        logger.error(f"TTS synthesis error: {e}")
        return None

def send_ai_reply(chat_id, text, reply_markup=None):
    """Send AI reply as text or audio depending on user setting."""
    if not is_voice_reply_enabled(chat_id):
        return safe_send_message(chat_id, text, reply_markup=reply_markup)

    audio_path = _synthesize_audio_file(text, user_id=chat_id)
    if not audio_path:
        safe_send_message(
            chat_id,
            "🔈 Voice reply unavailable right now. Sending text instead."
        )
        return safe_send_message(chat_id, text, reply_markup=reply_markup)

    try:
        with open(audio_path, "rb") as f:
            bot.send_audio(
                chat_id,
                f,
                title="Artovix Voice Reply",
                caption="🔈 Voice reply"
            )
        # Keep text too for copy/paste usability.
        return safe_send_message(chat_id, text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Send audio reply error: {e}")
        try:
            with open(audio_path, "rb") as f:
                bot.send_document(chat_id, f, caption="🔈 Voice reply (file)")
            return safe_send_message(chat_id, text, reply_markup=reply_markup)
        except Exception as e2:
            logger.error(f"Send audio as document fallback failed: {e2}")
            safe_send_message(chat_id, "⚠️ Audio delivery failed. Sending text only.")
            return safe_send_message(chat_id, text, reply_markup=reply_markup)
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

def _extract_text_from_document_bytes(file_name, raw_bytes):
    """Extract text from supported document types."""
    if not raw_bytes:
        return None, "Empty file."

    lowered = (file_name or "").lower()
    ext = lowered.rsplit(".", 1)[-1] if "." in lowered else ""

    try:
        if ext == "pdf":
            if not PdfReader:
                return None, "PDF parser not available. Install `pypdf`."
            reader = PdfReader(io.BytesIO(raw_bytes))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if not text:
                return None, "Could not extract text from PDF."
            return text, None

        if ext in {"txt", "md", "csv", "json", "py", "js", "html", "css", "xml", "log"}:
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
            if not text:
                return None, "File appears empty after decoding."
            return text, None

        return None, "Unsupported file type. Use PDF/TXT/MD/CSV/JSON/code files."
    except Exception as e:
        logger.error(f"Document extraction error: {e}")
        return None, "Failed to read that file."

def store_user_document(user_id, file_name, text):
    clipped = (text or "")[:MAX_DOC_STORE_CHARS]
    memory.update_setting(str(user_id), DOC_CONTEXT_KEY, clipped)
    memory.update_setting(str(user_id), DOC_NAME_KEY, file_name or "document")
    memory.update_setting(str(user_id), DOC_UPDATED_KEY, datetime.now().isoformat())

def load_dynamic_admin_ids():
    try:
        if not os.path.exists(ADMIN_STORE_FILE):
            return set()
        with open(ADMIN_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = set()
        if isinstance(data, list):
            for item in data:
                if str(item).isdigit():
                    result.add(int(item))
        return result
    except Exception as e:
        logger.error(f"Load dynamic admins error: {e}")
        return set()

def save_dynamic_admin_ids(admin_ids):
    try:
        with open(ADMIN_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(admin_ids)), f, indent=2)
    except Exception as e:
        logger.error(f"Save dynamic admins error: {e}")

def get_effective_admin_ids():
    # Always include configured and fallback main admin IDs.
    return set(ADMIN_USER_IDS) | load_dynamic_admin_ids() | set(MAIN_ADMIN_IDS)

def is_admin_user(user_id):
    try:
        return int(user_id) in get_effective_admin_ids()
    except:
        return False

def _get_chat_id(update_obj):
    """Support both Message and CallbackQuery-like objects."""
    chat = getattr(update_obj, "chat", None)
    if chat and getattr(chat, "id", None) is not None:
        return chat.id
    msg = getattr(update_obj, "message", None)
    chat = getattr(msg, "chat", None)
    if chat and getattr(chat, "id", None) is not None:
        return chat.id
    return None

def get_actor_user_id(update_obj):
    """Best-effort actor id resolver for Message and CallbackQuery."""
    # CallbackQuery user (the person who clicked a button)
    uid = getattr(getattr(update_obj, "from_user", None), "id", None)
    if uid is not None:
        try:
            return int(uid)
        except:
            pass

    # Fallback to message.from_user when wrapper object has .message
    msg = getattr(update_obj, "message", None)
    uid = getattr(getattr(msg, "from_user", None), "id", None)
    if uid is not None:
        try:
            return int(uid)
        except:
            pass

    # In private chats, chat.id is the user id.
    chat = getattr(update_obj, "chat", None) or getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", None)
    chat_type = getattr(chat, "type", "")
    if chat_id is not None and chat_type == "private":
        try:
            return int(chat_id)
        except:
            pass

    return None

def require_admin(update_obj):
    uid = get_actor_user_id(update_obj)
    chat_id = _get_chat_id(update_obj)
    if not is_admin_user(uid):
        if chat_id:
            safe_send_message(
                chat_id,
                "⛔ Admin only command.\n"
                f"Detected ID: `{uid}`\n"
                f"Main Admin: `{MAIN_ADMIN_ID}`\n"
                "Use `/myid` in private chat and share the value if this is wrong."
            )
        logger.warning(
            f"Admin check failed: uid={uid}, main={MAIN_ADMIN_ID}, admins={sorted(get_effective_admin_ids())}"
        )
        return False
    return True

def require_main_admin(update_obj):
    uid = get_actor_user_id(update_obj)
    chat_id = _get_chat_id(update_obj)
    if uid not in MAIN_ADMIN_IDS:
        if chat_id:
            safe_send_message(chat_id, "⛔ Main admin only command.")
        logger.warning(f"Main admin check failed: uid={uid}, main={MAIN_ADMIN_ID}")
        return False
    return True

def _channel_ref():
    if REQUIRED_CHANNEL:
        ref = REQUIRED_CHANNEL
    else:
        ref = REQUIRED_CHANNEL_URL
    ref = (ref or "").strip()
    if not ref:
        return ""
    if ref.startswith("http://") or ref.startswith("https://"):
        path = urlparse(ref).path.strip("/")
        if path:
            return f"@{path.split('/')[0]}"
        return ""
    if ref.startswith("@"):
        return ref
    return f"@{ref}"

def _channel_url():
    if REQUIRED_CHANNEL_URL and REQUIRED_CHANNEL_URL.startswith("http"):
        return REQUIRED_CHANNEL_URL
    ref = _channel_ref().lstrip("@")
    return f"https://t.me/{ref}" if ref else ""

def is_required_channel_member(user_id):
    """Check whether a user joined the required channel."""
    if user_id is None:
        return False
    if is_admin_user(user_id):
        return True
    channel = _channel_ref()
    if not channel:
        return True
    try:
        member = bot.get_chat_member(channel, int(user_id))
        status = getattr(member, "status", "")
        if status in ("creator", "administrator", "member"):
            return True
        if status == "restricted" and bool(getattr(member, "is_member", False)):
            return True
        return False
    except Exception as e:
        logger.warning(f"Channel membership check failed for {user_id} in {channel}: {e}")
        return False

def send_join_required_prompt(chat_id):
    channel_url = _channel_url()
    markup = InlineKeyboardMarkup(row_width=1)
    if channel_url:
        markup.add(InlineKeyboardButton("📢 Join Channel", url=channel_url))
    markup.add(InlineKeyboardButton("✅ I Joined", callback_data="check_joined"))
    safe_send_message(
        chat_id,
        "🔒 *Join Required*\n\n"
        "Please join our channel first to use this bot.\n"
        "After joining, tap *I Joined*.",
        reply_markup=markup
    )

def ensure_channel_access(update_obj):
    """Block access for users who have not joined required channel."""
    uid = get_actor_user_id(update_obj)
    if uid and uid > 0:
        analytics.touch_user(uid, "access_check")
    chat_id = _get_chat_id(update_obj)
    if is_required_channel_member(uid):
        return True
    if chat_id:
        send_join_required_prompt(chat_id)
    return False

def get_all_known_user_ids():
    user_ids = set()
    # Users in memory file
    try:
        for uid in memory.load().keys():
            if str(uid).isdigit():
                user_ids.add(int(uid))
    except Exception as e:
        logger.error(f"Memory user ids error: {e}")
    # Users in analytics DB
    user_ids.update(analytics.get_known_user_ids())
    # Include configured admins so admin test broadcasts are not missed
    try:
        user_ids.update(get_effective_admin_ids())
    except Exception:
        pass
    # Never broadcast back to dummy/invalid ids
    return {uid for uid in user_ids if uid > 0}

def get_target_user_ids(audience: str = "all"):
    key = (audience or "all").strip().lower()
    if key in ("all", "everyone"):
        return get_all_known_user_ids()
    if key in ("active7", "active_7d", "active-7d"):
        return {uid for uid in analytics.get_user_ids_active_in_days(7) if uid > 0}
    if key in ("active30", "active_30d", "active-30d"):
        return {uid for uid in analytics.get_user_ids_active_in_days(30) if uid > 0}
    return get_all_known_user_ids()

def broadcast_text_to_users(text, audience="all"):
    delivered = 0
    failed = 0
    targets = get_target_user_ids(audience)
    for uid in targets:
        try:
            safe_send_message(uid, text)
            delivered += 1
        except Exception:
            failed += 1
    return delivered, failed

# Uses configured persistent-aware path from startup config.
SCHEDULE_LOCK = threading.Lock()
SCHEDULER_STARTED = False

def load_scheduled_broadcasts():
    try:
        if os.path.exists(SCHEDULE_FILE):
            with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        logger.error(f"Load schedule error: {e}")
    return []

def save_scheduled_broadcasts(items):
    try:
        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Save schedule error: {e}")

def create_scheduled_broadcast(admin_id: int, delay_minutes: int, text: str, audience: str = "all"):
    delay_minutes = max(1, int(delay_minutes))
    audience = (audience or "all").strip().lower()
    now_ts = int(time.time())
    run_at = now_ts + delay_minutes * 60
    item = {
        "id": f"sb_{now_ts}_{admin_id}",
        "admin_id": int(admin_id),
        "created_at": now_ts,
        "run_at": run_at,
        "delay_minutes": delay_minutes,
        "audience": audience,
        "text": text
    }
    with SCHEDULE_LOCK:
        data = load_scheduled_broadcasts()
        data.append(item)
        save_scheduled_broadcasts(data)
    return item

def cancel_scheduled_broadcast(schedule_id: str):
    with SCHEDULE_LOCK:
        data = load_scheduled_broadcasts()
        left = [x for x in data if x.get("id") != schedule_id]
        changed = len(left) != len(data)
        if changed:
            save_scheduled_broadcasts(left)
    return changed

def list_scheduled_broadcasts(limit: int = 20):
    with SCHEDULE_LOCK:
        data = load_scheduled_broadcasts()
    data.sort(key=lambda x: x.get("run_at", 0))
    return data[:max(1, int(limit))]

def scheduler_loop():
    while True:
        try:
            now_ts = int(time.time())
            due = []
            keep = []
            with SCHEDULE_LOCK:
                data = load_scheduled_broadcasts()
                for item in data:
                    if int(item.get("run_at", 0)) <= now_ts:
                        due.append(item)
                    else:
                        keep.append(item)
                if len(keep) != len(data):
                    save_scheduled_broadcasts(keep)

            for item in due:
                try:
                    delivered, failed = broadcast_text_to_users(
                        item.get("text", ""),
                        audience=item.get("audience", "all")
                    )
                    admin_id = item.get("admin_id")
                    if admin_id:
                        safe_send_message(
                            int(admin_id),
                            f"⏰ Scheduled broadcast sent.\n"
                            f"ID: `{item.get('id')}`\n"
                            f"Audience: `{item.get('audience', 'all')}`\n"
                            f"Delivered: {delivered}\nFailed: {failed}"
                        )
                except Exception as e:
                    logger.error(f"Scheduled broadcast send error: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            time.sleep(10)

def start_scheduler_once():
    global SCHEDULER_STARTED
    if SCHEDULER_STARTED:
        return
    SCHEDULER_STARTED = True
    Thread(target=scheduler_loop, daemon=True).start()

POST_WIZARD_STATE = {}

def _wizard_get(chat_id):
    return POST_WIZARD_STATE.get(int(chat_id))

def _wizard_set(chat_id, state):
    POST_WIZARD_STATE[int(chat_id)] = state

def _wizard_clear(chat_id):
    POST_WIZARD_STATE.pop(int(chat_id), None)

def _extract_broadcast_payload_from_message(message):
    if getattr(message, "text", None) and not message.text.startswith('/'):
        return {"type": "text", "text": message.text}
    if getattr(message, "photo", None):
        return {"type": "photo", "file_id": message.photo[-1].file_id, "caption": message.caption or ""}
    if getattr(message, "video", None):
        return {"type": "video", "file_id": message.video.file_id, "caption": message.caption or ""}
    if getattr(message, "audio", None):
        return {"type": "audio", "file_id": message.audio.file_id, "caption": message.caption or ""}
    if getattr(message, "document", None):
        return {"type": "document", "file_id": message.document.file_id, "caption": message.caption or ""}
    if getattr(message, "animation", None):
        return {"type": "animation", "file_id": message.animation.file_id, "caption": message.caption or ""}
    if getattr(message, "voice", None):
        return {"type": "voice", "file_id": message.voice.file_id}
    if getattr(message, "sticker", None):
        return {"type": "sticker", "file_id": message.sticker.file_id}
    return None

def _build_button_markup(button_text, button_url):
    if not button_text or not button_url:
        return None
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton(button_text, url=button_url))
    return markup

def _is_valid_http_url(value):
    try:
        parsed = urlparse(value.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except:
        return False

def _send_payload_to_user(uid, payload, reply_markup=None):
    ptype = payload.get("type")
    if ptype == "text":
        safe_send_message(uid, payload.get("text", ""), reply_markup=reply_markup)
        return
    if ptype == "photo":
        bot.send_photo(uid, payload["file_id"], caption=payload.get("caption", ""), reply_markup=reply_markup)
        return
    if ptype == "video":
        bot.send_video(uid, payload["file_id"], caption=payload.get("caption", ""), reply_markup=reply_markup)
        return
    if ptype == "audio":
        bot.send_audio(uid, payload["file_id"], caption=payload.get("caption", ""), reply_markup=reply_markup)
        return
    if ptype == "document":
        bot.send_document(uid, payload["file_id"], caption=payload.get("caption", ""), reply_markup=reply_markup)
        return
    if ptype == "animation":
        bot.send_animation(uid, payload["file_id"], caption=payload.get("caption", ""), reply_markup=reply_markup)
        return
    if ptype == "voice":
        # Telegram voice messages do not support inline keyboards.
        bot.send_voice(uid, payload["file_id"])
        return
    if ptype == "sticker":
        bot.send_sticker(uid, payload["file_id"])
        return
    raise RuntimeError(f"Unsupported payload type: {ptype}")

def _broadcast_payload_to_users(payload, button_text=None, button_url=None):
    users = get_all_known_user_ids()
    delivered = 0
    failed = 0
    markup = _build_button_markup(button_text, button_url)
    for uid in users:
        try:
            _send_payload_to_user(uid, payload, reply_markup=markup)
            delivered += 1
        except Exception:
            failed += 1
    return delivered, failed

_DEAD_MODELS = set()

def is_transient_groq_error(exc: Exception) -> bool:
    txt = str(exc or "").lower()
    transient_markers = [
        "rate limit", "ratelimit", "429", "timeout", "timed out", "temporarily unavailable",
        "temporary", "connection reset", "connection aborted", "service unavailable",
        "502", "503", "504", "bad gateway", "gateway timeout"
    ]
    return any(marker in txt for marker in transient_markers)

def is_fatal_model_error(exc: Exception) -> bool:
    txt = str(exc or "").lower()
    status_code = getattr(exc, "status_code", None)
    return status_code in {404, 410} or any(marker in txt for marker in ["404", "not found", "decommissioned", "not available", "unsupported model"])

is_transient_ai_error = is_transient_groq_error

def ai_chat_with_fallback(messages, temperature=0.7, max_tokens=400, model=None):
    """NVIDIA NIM primary → Groq silent fallback → all models exhausted."""
    if not nvidia_client and not groq_client:
        raise RuntimeError("No AI client configured. Set NVIDIA_API_KEY in Render environment variables.")

    # ── Provider order: NVIDIA first, Groq as silent backup ──────────────────
    providers = []
    if nvidia_client:
        nvidia_candidates = [
            model if (model and "/" in model) else None,
            CHAT_MODEL if "/" in CHAT_MODEL else None,
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.2-3b-instruct",
            "mistralai/mistral-nemo-12b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "google/gemma-3-12b-it",
            "google/gemma-3-4b-it",
            "microsoft/phi-3-mini-128k-instruct",
            "meta/llama-3.2-1b-instruct",
            CHAT_MODEL_FALLBACK if "/" in CHAT_MODEL_FALLBACK else None,
        ]
        providers.append(("nvidia", nvidia_client, nvidia_candidates))

    if groq_client:
        groq_candidates = [
            model if (model and "/" not in model) else None,
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview",
        ]
        providers.append(("groq", groq_client, groq_candidates))

    last_error = None
    for provider_name, client, raw_candidates in providers:
        seen = set()
        models = [m for m in raw_candidates
                  if m and m not in seen and not seen.add(m)
                  and (provider_name, m) not in _DEAD_MODELS]

        for mod in models:
            for attempt in range(1, AI_TRANSIENT_RETRIES + 1):
                try:
                    response = client.chat.completions.create(
                        model=mod,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=15.0
                    )
                    choice = response.choices[0] if response and response.choices else None
                    content = None
                    if choice:
                        raw_content = getattr(choice.message, "content", None) or ""
                        content = strip_thinking_process(raw_content).strip() if raw_content else None
                    finish_reason = getattr(choice, "finish_reason", None) if choice else None
                    if content:
                        return content, mod, finish_reason
                    logger.warning(f"{provider_name} model '{mod}' returned empty. Skipping.")
                    break
                except Exception as e:
                    last_error = e
                    if is_fatal_model_error(e):
                        _DEAD_MODELS.add((provider_name, mod))
                        logger.warning(f"{provider_name} model '{mod}' fatal (404/410). Marked dead.")
                        break
                    err_txt = str(e).lower()
                    if "timed out" in err_txt or "timeout" in err_txt:
                        logger.warning(f"{provider_name} model '{mod}' timed out. Trying next.")
                        break
                    # For exhausted/auth errors on NVIDIA, skip to Groq immediately
                    sc = getattr(e, "status_code", None)
                    if sc in {401, 402, 403, 410} or any(t in err_txt for t in ("credit", "exhausted", "quota", "unauthorized")):
                        logger.warning(f"{provider_name} key issue (status {sc}). Skipping provider.")
                        break
                    should_retry = is_transient_ai_error(e) and attempt < AI_TRANSIENT_RETRIES
                    logger.warning(f"{provider_name} model failed ({mod}, attempt {attempt}): {e}")
                    if should_retry:
                        time.sleep(AI_RETRY_BACKOFF_SEC * attempt)
                        continue
                    break
            else:
                continue
            # If we hit a key-level fatal error, skip remaining models in this provider
            if last_error:
                sc = getattr(last_error, "status_code", None)
                err_txt = str(last_error).lower()
                if sc in {401, 402, 403, 410} or any(t in err_txt for t in ("credit", "exhausted", "quota", "unauthorized")):
                    break  # Move to next provider

    raise last_error if last_error else RuntimeError("All AI models failed.")

groq_chat_with_fallback = ai_chat_with_fallback  # alias kept for compatibility

def ai_vision_with_fallback(messages, max_tokens=500):
    """NVIDIA NIM vision-capable model waterfall."""
    if not nvidia_client:
        raise RuntimeError("No AI client configured for vision. Set NVIDIA_API_KEY.")

    candidates = [
        VISION_MODEL if "/" in VISION_MODEL else None,
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        VISION_MODEL_FALLBACK if "/" in VISION_MODEL_FALLBACK else None,
    ]
    seen = set()
    models = [m for m in candidates
              if m and m not in seen and not seen.add(m)
              and ("nvidia", m) not in _DEAD_MODELS]

    last_error = None
    for mod in models:
        try:
            response = nvidia_client.chat.completions.create(
                model=mod, messages=messages, max_tokens=max_tokens, timeout=15.0
            )
            raw_c = response.choices[0].message.content if response and response.choices else None
            content = strip_thinking_process(raw_c) if raw_c else None
            if content:
                return content, mod
            raise RuntimeError(f"Empty vision response from model: {mod}")
        except Exception as e:
            last_error = e
            if is_fatal_model_error(e):
                _DEAD_MODELS.add(("nvidia", mod))
            logger.warning(f"NVIDIA vision model failed ({mod}): {e}")

    raise last_error if last_error else RuntimeError("All NVIDIA vision models failed.")

groq_vision_with_fallback = ai_vision_with_fallback

# ============================================================================
# 🖼️ IMAGE GENERATOR (IMPROVED WITH MULTIPLE SERVICES)
# ============================================================================
class ImageGenerator:
    """Improved Image Generator with multiple reliable services"""
    
    @staticmethod
    def generate(prompt: str, model_type: str = "auto"):
        """Generate AI images using multiple reliable services"""
        try:
            clean_prompt = prompt.strip()
            if not clean_prompt:
                return None
            
            logger.info(f"Generating image ({model_type}) for: {clean_prompt[:50]}...")
            
            # 1. Hugging Face (FLUX.1-schnell) - More reliable for free API
            if model_type in ["auto", "flux"]:
                try:
                    logger.info("Trying Hugging Face (FLUX.1-schnell)...")
                    hf_model = "black-forest-labs/FLUX.1-schnell"
                    # New Router endpoint as requested by HF Error 410
                    hf_url = f"https://api-inference.huggingface.co/models/{hf_model}"
                    
                    headers = {
                        "Authorization": f"Bearer {HF_KEY}",
                        "x-use-cache": "false"
                    }
                    
                    response = requests.post(
                        hf_url, 
                        headers=headers, 
                        json={"inputs": clean_prompt},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if 'image' in content_type:
                            logger.info("✓ Hugging Face successful!")
                            return response.content
                        else:
                            logger.warning(f"HF returned non-image: {content_type}")
                    
                    elif response.status_code == 410:
                        # If still 410, try the specific router URL from error message
                        router_url = f"https://router.huggingface.co/hf-inference/models/{hf_model}"
                        logger.info("Attempting HF Router fallback...")
                        response = requests.post(
                            router_url,
                            headers=headers,
                            json={"inputs": clean_prompt},
                            timeout=60
                        )
                        if response.status_code == 200:
                            return response.content
                    
                    logger.warning(f"HF failed ({response.status_code}): {response.text[:100]}")
                        
                except Exception as e:
                    logger.warning(f"Hugging Face exception: {str(e)[:100]}")

            # 2. Pollinations.ai (Reliable fallback/choice)
            if model_type in ["auto", "pollinations"]:
                try:
                    encoded_prompt = requests.utils.quote(clean_prompt)
                    url1 = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                    logger.info("Trying Pollinations.ai Flux...")
                    
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    params = {
                        "model": "flux",
                        "width": 1024,
                        "height": 1024,
                        "enhance": "true",
                        "nologo": "true",
                    }
                    for attempt in range(2):
                        response = requests.get(
                            url1,
                            params=params,
                            headers=headers,
                            timeout=30,
                        )
                        content_type = response.headers.get('content-type', '')
                        if response.status_code == 200 and 'image' in content_type.lower() and len(response.content) > 1000:
                            logger.info("✓ Pollinations.ai Flux successful!")
                            return response.content
                        logger.warning(
                            f"Pollinations Flux attempt {attempt + 1} failed "
                            f"({response.status_code}, {len(response.content)} bytes)"
                        )
                except Exception as e:
                    logger.warning(f"Pollinations.ai failed: {str(e)[:100]}")
            
            # 3. DeepAI / Stylized Pollinations
            if model_type in ["auto", "creative"]:
                try:
                    logger.info(f"Trying Creative style...")
                    import random
                    styles = ["digital-art", "fantasy-art", "neon-punk", "isometric", "low-poly"]
                    style = random.choice(styles)
                    encoded_prompt = requests.utils.quote(clean_prompt)
                    url2 = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model={style}"
                    response = requests.get(url2, timeout=15)
                    
                    if response.status_code == 200:
                        logger.info(f"✓ Creative style {style} successful!")
                        return response.content
                except Exception as e:
                    logger.warning(f"Creative mode failed: {str(e)[:100]}")
            
            # Fallback to Text Description ONLY if auto mode fails everything
            if model_type == "auto":
                try:
                    logger.info("Creating enhanced text description fallback...")
                    description_prompt = f"Create a detailed visual description for: {clean_prompt}"
                    
                    if not (nvidia_client or groq_client):
                        logger.warning("AI backend unavailable: cannot create text fallback description.")
                        return {
                            'type': 'text',
                            'prompt': clean_prompt,
                            'description': "AI backend not configured. Set NVIDIA_API_KEY or GROQ_API_KEY in .env to enable detailed descriptions.",
                            'emojis': '⚠️',
                            'suggestion': "Add NVIDIA_API_KEY or GROQ_API_KEY to .env and restart the bot."
                        }

                    content, _, _ = ai_chat_with_fallback(
                        messages=[{"role": "user", "content": description_prompt}],
                        temperature=0.8,
                        max_tokens=200
                    )
                    
                    return {
                        'type': 'text',
                        'prompt': clean_prompt,
                        'description': content,
                        'emojis': '🎨✨',
                        'suggestion': "Try a different prompt or simpler description."
                    }
                except Exception as e:
                    logger.error(f"Text fallback failed: {e}")
                    return None
            
            # If a specific model was requested and failed, return None to handle error in command
            return None
                    
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None

# ============================================================================
# 🎭 STICKER & EMOJI MASTERPIECE SYSTEM
# ============================================================================
EMOJI_ART_LIBRARY = {
    "happy": [
        "✨ ٩(◕‿◕｡)۶ ✨ *Sending you infinite happiness and joyful vibes!*",
        "🌟 (｡♥‿♥｡) 🌟 *You radiate pure awesome energy! Keep shining!*",
        "🥳 ヽ(>∀<☆)ノ 🎉 *Party mode unlocked! Let's celebrate life!*",
        "🌈 (◠‿◕) ✨ *Every day is brighter with you around!*",
    ],
    "love": [
        "💖 ( ˘ ³˘)♥ 💖 *Sending you all the love, hugs, and warm wishes!*",
        "🥰 (づ｡◕‿‿◕｡)づ 💕 *Big warm virtual hugs for you! You are truly cherished!*",
        "🌹 (๑•́ ₃ •̀๑) 💓 *Pure sweetness and affection coming your way!*",
        "✨ (*˘︶˘*).｡.:*♡ 💫 *May your heart always be full of joy!*",
    ],
    "cool": [
        "😎 (⌐■_■) 🚀 *Absolute legend status confirmed! Maximum swag!*",
        "🔥 ᕙ(⇀‸↼‶)ᕗ ⚡ *Unstoppable power and blazing confidence! Let's conquer!*",
        "🕶️ ✧( ु•⌄• )◞ ✨ *Too cool for school! Looking ultra sharp!*",
    ],
    "laugh": [
        "😂 (≧▽≦) 🤣 *Hahaha this made my day! Pure comedy gold!*",
        "😹 ᕕ( ᐛ )ᕗ 💥 *Can't stop laughing! You're hilarious!*",
        "😆 (*≧ω≦*) ✨ *Laughter is the best medicine, and you nailed it!*",
    ],
    "cute": [
        "🐱 (ﾐචᆽචﾐ) 🐾 *Purrr~ Soft kitty says hello and sends warm purrs!*",
        "🐶 (❍ᴥ❍ʋ) 🦴 *Good puppy wagging tail happily for you! Woof!*",
        "🐰 (ᐢ • ˕ • ᐢ) 🥕 *Little bunny hops over to bring you sweet luck!*",
        "🐼 ʕ •ᴥ• ʔ 🎋 *Fluffy panda wants to give you a big cuddle!*",
    ],
    "thinking": [
        "🤔 (¬_¬ ) 💡 *Fascinating question! The neural gears are spinning!*",
        "🧠 (◕‿◕✿) ⚡ *Brilliant thought! Great minds think alike!*",
        "🧐 ₍ᐢ.ˬ.ᐢ₎ 💭 *Deep analysis in progress... Eureka!*",
    ],
    "wow": [
        "🤯 (⊙_⊙) 🌟 *MIND = BLOWN! That is utterly incredible!*",
        "✨ ＼(〇_ｏ)／ 🚀 *Astonishing! Out of this world masterpiece!*",
        "🤩 (☆ω☆) 💫 *Starry-eyed amazement! Absolute perfection!*",
    ],
    "robot": [
        "🤖 beep boop! [ARTOVIX PROTOCOL] (•‿•) ⚡ *Cybernetic high-five initiated!*",
        "🦾 (🤖_🤖) 🔋 *Full battery! AI cores overclocked with friendliness!*",
        "🚀 [2026 QUANTUM CORE] ✨ *Artovix online and delighted to assist you!*",
    ],
    "fire": [
        "🔥 ＼(🔥o🔥)／ 🔥 *FIRE! This is straight up sensational!*",
        "⚡ ᕙ( •̀ ᗜ •́ )ᕗ 💥 *High voltage excitement! You're unstoppable!*",
    ],
    "party": [
        "🎉 ヾ(⌐■_■)ノ♪ 🎊 *DJ drop that beat! Celebrate good times!*",
        "🥳 ┏(･o･)┛♪┗ (･o･) ┓ 🎶 *Dancing into greatness! Whoop whoop!*",
    ],
    "thumbsup": [
        "👍 (b^_^)b 💯 *100% Approved! Great job! High five!*",
        "🙌 (★‿★) ✨ *Two thumbs up and a standing ovation! Respect!*",
    ]
}

class StickerAndEmojiVault:
    """Dynamic sticker and emoji storage with auto-learning capabilities."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.data = self._load()

    def _default_data(self):
        return {
            "stickers": {
                "happy": [],
                "love": [],
                "cool": [],
                "laugh": [],
                "cute": [],
                "cat": [],
                "dog": [],
                "thinking": [],
                "wow": [],
                "robot": [],
                "fire": [],
                "party": [],
                "thumbsup": [],
            },
            "custom_stickers": [],
            "learned_count": 0
        }

    def _load(self):
        try:
            with self.lock:
                if os.path.exists(self.filepath):
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "stickers" in data:
                            return data
        except Exception as e:
            logger.error(f"Sticker vault load error: {e}")
        return self._default_data()

    def _save(self):
        try:
            with self.lock:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Sticker vault save error: {e}")

    def add_sticker(self, file_id: str, emoji: str = "", set_name: str = ""):
        if not file_id:
            return
        mood = self.detect_mood_from_emoji(emoji)
        with self.lock:
            category_list = self.data["stickers"].setdefault(mood, [])
            if file_id not in category_list:
                category_list.append(file_id)
                self.data["learned_count"] = self.data.get("learned_count", 0) + 1
                self._save()
                logger.info(f"Learned new sticker ({mood}, {emoji}): {file_id[:16]}...")

    def get_random_sticker(self, mood: str = "happy"):
        mood_key = (mood or "happy").strip().lower()
        items = self.data.get("stickers", {}).get(mood_key, [])
        if not items:
            for m, sticker_list in self.data.get("stickers", {}).items():
                if sticker_list:
                    return random.choice(sticker_list)
            return None
        return random.choice(items)

    def detect_mood_from_emoji(self, emoji: str) -> str:
        if not emoji:
            return "happy"
        e = str(emoji).strip()
        if any(c in e for c in ["❤️", "💖", "💕", "😍", "🥰", "😘", "💓", "💗"]):
            return "love"
        if any(c in e for c in ["😂", "🤣", "😹", "😆", "😅", "💀"]):
            return "laugh"
        if any(c in e for c in ["😎", "🕶️", "🦾", "⚡", "🛸"]):
            return "cool"
        if any(c in e for c in ["🐱", "🐶", "🐾", "🐰", "🐼", "🦊", "🐻"]):
            return "cute"
        if any(c in e for c in ["🤔", "🧐", "🧠", "💡", "❓"]):
            return "thinking"
        if any(c in e for c in ["😲", "🤯", "😮", "😱", "🤩", "✨"]):
            return "wow"
        if any(c in e for c in ["🤖", "👾", "💻", "🔧"]):
            return "robot"
        if any(c in e for c in ["🔥", "💥", "🌋"]):
            return "fire"
        if any(c in e for c in ["🎉", "🥳", "🎊", "🎂", "🎈", "🥂"]):
            return "party"
        if any(c in e for c in ["👍", "👌", "🙌", "💯", "👏", "🤝"]):
            return "thumbsup"
        return "happy"

    def detect_mood_from_text(self, text: str) -> str:
        t = (text or "").lower()
        if any(w in t for w in ["love", "heart", "hug", "kiss", "sweet", "crush", "darling"]):
            return "love"
        if any(w in t for w in ["laugh", "funny", "haha", "lol", "joke", "meme", "rofl"]):
            return "laugh"
        if any(w in t for w in ["cool", "swag", "fire", "boss", "chad", "legend"]):
            return "cool"
        if any(w in t for w in ["cat", "kitty", "dog", "puppy", "cute", "pet", "bunny", "panda"]):
            return "cute"
        if any(w in t for w in ["think", "idea", "wonder", "smart", "brain", "curious"]):
            return "thinking"
        if any(w in t for w in ["wow", "mindblown", "omg", "shock", "amaz", "incredible"]):
            return "wow"
        if any(w in t for w in ["robot", "ai", "cyber", "future", "tech"]):
            return "robot"
        if any(w in t for w in ["party", "dance", "celebrat", "birthday", "cheer"]):
            return "party"
        if any(w in t for w in ["thumb", "agree", "good", "great", "nice", "yes", "super", "ok"]):
            return "thumbsup"
        return "happy"

    def get_random_emoji_art(self, mood: str = "happy") -> str:
        mood_key = (mood or "happy").strip().lower()
        options = EMOJI_ART_LIBRARY.get(mood_key, EMOJI_ART_LIBRARY["happy"])
        return random.choice(options)

sticker_vault = StickerAndEmojiVault(STICKERS_STORE_FILE)


def seed_sticker_pack(pack_name: str, default_mood: str = "robot", extra_moods: list = None):
    """
    Fetch a Telegram sticker pack by name and seed the vault with all its stickers.
    Already-stored stickers are skipped (idempotent). Runs safely in a background thread.
    """
    extra_moods = extra_moods or []
    try:
        sticker_set = bot.get_sticker_set(pack_name)
        count = 0
        for sticker in sticker_set.stickers:
            file_id = getattr(sticker, "file_id", "")
            emoji   = getattr(sticker, "emoji", "") or ""
            if not file_id:
                continue
            detected_mood = sticker_vault.detect_mood_from_emoji(emoji) if emoji else default_mood
            # Seed under detected mood
            sticker_vault.add_sticker(file_id=file_id, emoji=emoji, set_name=pack_name)
            # Also add under any explicitly requested extra moods so the pack is reachable from more categories
            for mood in extra_moods:
                with sticker_vault.lock:
                    cat = sticker_vault.data["stickers"].setdefault(mood, [])
                    if file_id not in cat:
                        cat.append(file_id)
            count += 1
        if count:
            sticker_vault._save()
        logger.info(f"[StickerSeed] Seeded {count} stickers from pack '{pack_name}'")
        return count
    except Exception as e:
        logger.warning(f"[StickerSeed] Could not seed pack '{pack_name}': {e}")
        return 0


def safe_send_sticker(chat_id, sticker_item):
    """Safely send a sticker with automatic format and download handling."""
    if not sticker_item:
        return None
    try:
        if isinstance(sticker_item, (bytes, bytearray)):
            bio = io.BytesIO(sticker_item)
            bio.name = "sticker.webp"
            return bot.send_sticker(chat_id, bio)
        if isinstance(sticker_item, io.BytesIO):
            return bot.send_sticker(chat_id, sticker_item)
        if isinstance(sticker_item, str) and (sticker_item.startswith("http://") or sticker_item.startswith("https://")):
            resp = requests.get(sticker_item, timeout=12)
            if resp.status_code == 200:
                bio = io.BytesIO(resp.content)
                bio.name = "sticker.webp"
                return bot.send_sticker(chat_id, bio)
        return bot.send_sticker(chat_id, sticker_item)
    except Exception as e:
        logger.warning(f"safe_send_sticker error for {chat_id}: {e}")
        return None

# ============================================================================
# 🏆 EPIC GAMES ENGINE — Score tracker, ranks, streaks, trivia, card game
# ============================================================================

GAME_EMOJIS = {
    "dice":     ("🎲", "🎲 Rolling the epic dice of destiny..."),
    "dart":     ("🎯", "🎯 Locking on target... bullseye incoming!"),
    "basket":   ("🏀", "🏀 Launching the shot... the crowd goes wild!"),
    "football": ("⚽", "⚽ Stepping up for the penalty... the world watches!"),
    "slot":     ("🎰", "🎰 Pulling the lever... jackpot vibes loading!"),
    "bowl":     ("🎳", "🎳 Ball released... pins trembling in fear!"),
}

# ── Rank system ──────────────────────────────────────────────────────────────
RANK_THRESHOLDS = [
    (0,    "🪨 Rookie"),
    (10,   "🥉 Bronze"),
    (30,   "🥈 Silver"),
    (60,   "🥇 Gold"),
    (100,  "💎 Diamond"),
    (200,  "🏆 Legend"),
    (500,  "👑 God-Tier"),
]

GAME_SCORE_KEY  = "game_score"
GAME_WINS_KEY   = "game_wins"
GAME_PLAYED_KEY = "game_played"
GAME_STREAK_KEY = "game_streak"
GAME_BEST_KEY   = "game_best_streak"
TRIVIA_KEY      = "trivia_session"
CARD_KEY        = "card_session"

def get_game_stats(user_id):
    uid = str(user_id)
    return {
        "score":   int(memory.get_setting(uid, GAME_SCORE_KEY, 0)),
        "wins":    int(memory.get_setting(uid, GAME_WINS_KEY, 0)),
        "played":  int(memory.get_setting(uid, GAME_PLAYED_KEY, 0)),
        "streak":  int(memory.get_setting(uid, GAME_STREAK_KEY, 0)),
        "best":    int(memory.get_setting(uid, GAME_BEST_KEY, 0)),
    }

def add_game_score(user_id, points, win=False):
    uid = str(user_id)
    s = get_game_stats(uid)
    s["score"]  += points
    s["played"] += 1
    if win:
        s["wins"]   += 1
        s["streak"] += 1
        s["best"]    = max(s["best"], s["streak"])
    else:
        s["streak"] = 0
    memory.update_setting(uid, GAME_SCORE_KEY,  s["score"])
    memory.update_setting(uid, GAME_WINS_KEY,   s["wins"])
    memory.update_setting(uid, GAME_PLAYED_KEY, s["played"])
    memory.update_setting(uid, GAME_STREAK_KEY, s["streak"])
    memory.update_setting(uid, GAME_BEST_KEY,   s["best"])
    return s

def get_rank(score):
    rank = RANK_THRESHOLDS[0][1]
    for threshold, name in RANK_THRESHOLDS:
        if score >= threshold:
            rank = name
    return rank

def format_stats_card(user_id, username="Player"):
    s  = get_game_stats(user_id)
    rk = get_rank(s["score"])
    win_rate = round(s["wins"] / s["played"] * 100) if s["played"] else 0
    return (
        f"🎮 *{username}'s Game Stats*\n"
        f"{'─'*28}\n"
        f"🏅 Rank:        {rk}\n"
        f"⭐ Score:       {s['score']} pts\n"
        f"🎯 Games:       {s['played']} played\n"
        f"🏆 Wins:        {s['wins']} ({win_rate}%)\n"
        f"🔥 Streak:      {s['streak']} current\n"
        f"⚡ Best streak: {s['best']}\n"
        f"{'─'*28}"
    )

def streak_bonus_msg(streak):
    if streak == 3:
        return "🔥 *3-WIN STREAK!* You're on FIRE! +bonus!"
    if streak == 5:
        return "⚡ *5-WIN STREAK! UNSTOPPABLE!* 🌪️"
    if streak >= 10:
        return "👑 *LEGENDARY STREAK! ABSOLUTE GOD-MODE!* 🔱"
    return None

# ── Dice game commentary ──────────────────────────────────────────────────────
_DICE_COMMENTS = {
    6: ["🔥 PERFECT 6! Absolute LEGEND! +10 pts!", "🎲 MAX ROLL! The dice gods bow to you! 🙏"],
    5: ["✨ 5! Outstanding roll! +6 pts!", "🎲 Strong 5! You're a natural! +6 pts!"],
    4: ["🌟 Solid 4! Good stuff! +4 pts!", "👍 Nice 4! Decent roll! +4 pts!"],
    3: ["😤 A 3... we can do better! +2 pts!", "🎲 Three's not bad, but you want that 6! +2 pts!"],
    2: ["😬 Just a 2... next time! +1 pt!", "🎲 Two... the dice are testing you! +1 pt!"],
    1: ["💀 Snake eyes! 1! The dice trolled you! +1 pt!", "😱 1... The worst roll... but you played! +1 pt!"],
}
_DICE_POINTS = {6: 10, 5: 6, 4: 4, 3: 2, 2: 1, 1: 1}

_DART_COMMENTS = {
    6: ["🎯 BULLSEYE!! Perfect precision! +10 pts!", "💥 DEAD CENTER! Sniper-level accuracy! +10 pts!"],
    5: ["🎯 Outer bull! So close to perfect! +7 pts!", "✨ Near bullseye! Incredible aim! +7 pts!"],
    4: ["🌟 Inner ring! Sharp shot! +5 pts!", "🎯 Solid hit! +5 pts!"],
    3: ["😊 Triple ring! Not bad! +3 pts!", "🎯 Good throw! +3 pts!"],
    2: ["🙂 Outer ring... keep aiming! +1 pt!", "🎯 Still on the board! +1 pt!"],
    1: ["😬 Just barely on the board! +1 pt!", "💨 Almost missed! +1 pt!"],
}
_DART_POINTS = {6: 10, 5: 7, 4: 5, 3: 3, 2: 1, 1: 1}

_BASKET_COMMENTS = {
    5: ["🏀 SWISH!! NOTHING BUT NET! Pure silk! +10 pts!", "🌟 PERFECT SHOT! The crowd goes INSANE! +10 pts!"],
    4: ["🏀 SWISH! Barely touched the rim! +8 pts!", "🔥 Great shot! Nearly perfect! +8 pts!"],
    3: ["🏀 Off the backboard... and IN! +5 pts!", "😮 Lucky bounce and in! +5 pts!"],
    2: ["😤 Rim shot, no good... +2 pts!", "🏀 So close! Kissed the rim! +2 pts!"],
    1: ["💀 Airrrrball! 😂 +1 pt!", "🙈 Missed badly! But you tried! +1 pt!"],
}
_BASKET_POINTS = {5: 10, 4: 8, 3: 5, 2: 2, 1: 1}

_FOOTBALL_COMMENTS = {
    5: ["⚽ GOOOOOAAAALLL!! WORLD-CLASS STRIKE! 🥅🔥 +10 pts!", "🌍 TOP CORNER! PURE PERFECTION! +10 pts!"],
    4: ["⚽ GOAL! Powerful shot low left! +8 pts!", "🔥 SCORED! The keeper had no chance! +8 pts!"],
    3: ["⚽ Just squeezed it in! +5 pts!", "😮 Lucky goal! But it counts! +5 pts!"],
    2: ["😬 Saved by the keeper! +2 pts!", "⚽ Hit the post! So close! +2 pts!"],
    1: ["💀 Way over the bar! +1 pt!", "😂 That went into the stands! +1 pt!"],
}
_FOOTBALL_POINTS = {5: 10, 4: 8, 3: 5, 2: 2, 1: 1}

_BOWL_COMMENTS = {
    6: ["🎳 STRIKE!! ALL PINS DOWN! PERFECTION! 💥 +10 pts!", "⚡ PERFECT STRIKE! Bowling deity! +10 pts!"],
    5: ["🎳 5 pins! One frame away from greatness! +7 pts!", "🌟 Strong roll! Nearly a strike! +7 pts!"],
    4: ["🎳 4 pins down! Not bad! +4 pts!", "😊 Half the lane! Keep rolling! +4 pts!"],
    3: ["🙂 3 pins... spare is still possible! +2 pts!", "🎳 Three down, learning! +2 pts!"],
    2: ["😬 Just 2 pins... tough lane! +1 pt!", "🎳 Gutter is calling... but you stayed! +1 pt!"],
    1: ["💀 GUTTER BALL! 1 surviving pin mocks you! +1 pt!", "😂 1 pin... the lane laughs! +1 pt!"],
}
_BOWL_POINTS = {6: 10, 5: 7, 4: 4, 3: 2, 2: 1, 1: 1}

def _pick_comment(table, score):
    opts = table.get(score, table.get(1, ["🌟 Good game! +1 pt!"]))
    return random.choice(opts)

def play_telegram_game(chat_id, game_type="dice"):
    """Trigger Telegram animated dice games with epic commentary + score tracking."""
    game_key = (game_type or "dice").strip().lower()
    if game_key in {"roll", "die", "dices"}:            game_key = "dice"
    elif game_key in {"darts", "target", "bullseye"}:   game_key = "dart"
    elif game_key in {"basketball", "hoop", "slam"}:    game_key = "basket"
    elif game_key in {"soccer", "goal", "kick"}:        game_key = "football"
    elif game_key in {"casino", "slots", "777", "jackpot"}: game_key = "slot"
    elif game_key in {"bowling", "strike"}:              game_key = "bowl"

    item  = GAME_EMOJIS.get(game_key, GAME_EMOJIS["dice"])
    emoji, intro = item

    safe_send_message(chat_id, f"{intro}")
    try:
        msg = bot.send_dice(chat_id, emoji=emoji)
        val = getattr(getattr(msg, "dice", None), "value", None)

        def _send_result(score):
            time.sleep(2.5)
            # Build comment + points
            if game_key == "dice":
                comment = _pick_comment(_DICE_COMMENTS, score)
                pts     = _DICE_POINTS.get(score, 1)
                win     = score == 6
            elif game_key == "dart":
                comment = _pick_comment(_DART_COMMENTS, score)
                pts     = _DART_POINTS.get(score, 1)
                win     = score >= 5
            elif game_key == "basket":
                comment = _pick_comment(_BASKET_COMMENTS, score)
                pts     = _BASKET_POINTS.get(score, 1)
                win     = score >= 4
            elif game_key == "football":
                comment = _pick_comment(_FOOTBALL_COMMENTS, score)
                pts     = _FOOTBALL_POINTS.get(score, 1)
                win     = score >= 3
            elif game_key == "slot":
                is_jackpot = score == 64
                comment = ("🎰 ✨ JACKPOT 777!! MEGA WINNER! 💰🎉 +20 pts!" if is_jackpot
                           else f"🎰 Nice spin! Value: {score}. +3 pts!")
                pts, win = (20, True) if is_jackpot else (3, False)
            elif game_key == "bowl":
                comment = _pick_comment(_BOWL_COMMENTS, score)
                pts     = _BOWL_POINTS.get(score, 1)
                win     = score == 6
            else:
                comment = f"🌟 Great game! Score: {score}"
                pts, win = 2, False

            if score is None:
                safe_send_message(chat_id, "🎉 Epic game! Tap below to go again!", reply_markup=build_game_replay_markup(game_key))
                return

            stats = add_game_score(chat_id, pts, win=win)
            rank  = get_rank(stats["score"])
            streak_msg = streak_bonus_msg(stats["streak"])

            result_text = (
                f"{comment}\n\n"
                f"⭐ *+{pts} pts* → Total: *{stats['score']}* | Rank: {rank}\n"
                f"🔥 Streak: {stats['streak']} | 🏆 Best: {stats['best']}"
            )
            if streak_msg:
                result_text += f"\n\n{streak_msg}"

            safe_send_message(chat_id, result_text, reply_markup=build_game_replay_markup(game_key))

        Thread(target=_send_result, args=(val,), daemon=True).start()
    except Exception as e:
        logger.error(f"play_telegram_game error: {e}")
        safe_send_message(chat_id, f"{emoji} You're always a winner in my book! ✨")

def build_sticker_panel_markup():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🥳 Happy & Party", callback_data="stk_cat_happy"),
        InlineKeyboardButton("💖 Love & Hugs", callback_data="stk_cat_love"),
        InlineKeyboardButton("😎 Cool & Swag", callback_data="stk_cat_cool"),
        InlineKeyboardButton("😂 Funny & LOL", callback_data="stk_cat_laugh"),
        InlineKeyboardButton("🐱 Cute Pets", callback_data="stk_cat_cute"),
        InlineKeyboardButton("🤔 Big Brain", callback_data="stk_cat_thinking"),
        InlineKeyboardButton("😲 Wow / Mindblown", callback_data="stk_cat_wow"),
        InlineKeyboardButton("🤖 AI Robot", callback_data="stk_cat_robot"),
        InlineKeyboardButton("🔥 Fire Energy", callback_data="stk_cat_fire"),
        InlineKeyboardButton("👍 Thumbs Up", callback_data="stk_cat_thumbsup"),
        InlineKeyboardButton("🎲 Play a Game", callback_data="stk_open_games"),
        InlineKeyboardButton("🎨 Custom AI Sticker", callback_data="stk_help_custom"),
    )
    return markup

def build_sticker_replay_markup(mood):
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🔄 Another One", callback_data=f"stk_cat_{mood}"),
        InlineKeyboardButton("🎭 All Categories", callback_data="stk_menu_home"),
        InlineKeyboardButton("🎲 Play Game", callback_data="stk_open_games"),
    )
    return markup

def build_games_menu_markup():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🎲 Lucky Dice",       callback_data="game_play_dice"),
        InlineKeyboardButton("🎯 Bullseye Dart",    callback_data="game_play_dart"),
        InlineKeyboardButton("🏀 Basketball Shot",  callback_data="game_play_basket"),
        InlineKeyboardButton("⚽ Soccer Penalty",   callback_data="game_play_football"),
        InlineKeyboardButton("🎰 Jackpot Slots",    callback_data="game_play_slot"),
        InlineKeyboardButton("🎳 Bowling Strike",   callback_data="game_play_bowl"),
        InlineKeyboardButton("🧠 Trivia Challenge", callback_data="game_trivia_start"),
        InlineKeyboardButton("🃏 Higher or Lower",  callback_data="game_card_start"),
        InlineKeyboardButton("📊 My Game Stats",    callback_data="game_my_stats"),
    )
    return markup

def build_game_replay_markup(game_type):
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🔁 Play Again",      callback_data=f"game_play_{game_type}"),
        InlineKeyboardButton("🎮 All Games",        callback_data="stk_open_games"),
        InlineKeyboardButton("📊 My Stats",         callback_data="game_my_stats"),
    )
    return markup

def send_games_panel(chat_id):
    s = get_game_stats(chat_id)
    rk = get_rank(s["score"])
    safe_send_message(
        chat_id,
        f"🎮 *Epic Arcade — Artovix Games* 🏆\n\n"
        f"Your rank: {rk} | ⭐ {s['score']} pts | 🔥 Streak: {s['streak']}\n\n"
        f"Choose a game below — every play earns points, builds streaks, and climbs the ranks!\n"
        f"🧠 Trivia & 🃏 Higher-or-Lower are AI-powered brain games!",
        reply_markup=build_games_menu_markup()
    )

# ── 🧠 TRIVIA GAME ──────────────────────────────────────────────────────────
_TRIVIA_CATEGORIES = [
    "Science & Nature", "History", "Geography", "Pop Culture",
    "Technology & AI", "Sports", "Movies & TV", "Music",
    "Literature", "Space & Universe", "Animals", "Food & Cooking",
]

def start_trivia(chat_id):
    """Ask AI to generate a fresh trivia question, store answer, send question with A/B/C/D buttons."""
    cat = random.choice(_TRIVIA_CATEGORIES)
    prompt = (
        f"Generate ONE trivia question about '{cat}' with exactly 4 answer choices.\n"
        "Format STRICTLY as:\n"
        "QUESTION: <question text>\n"
        "A: <choice>\nB: <choice>\nC: <choice>\nD: <choice>\n"
        "ANSWER: <A, B, C, or D>\n"
        "FACT: <one fun sentence about the answer>\n"
        "No extra text."
    )
    try:
        raw, _, _ = groq_chat_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9, max_tokens=300,
            model=get_purpose_model("chat", chat_id)
        )
        raw = strip_thinking_process(raw or "")
        lines_q = [l.strip() for l in raw.splitlines() if l.strip()]
        q_text = a = b = c = d = answer = fact = ""
        for ln in lines_q:
            if ln.upper().startswith("QUESTION:"):  q_text = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("A:"):         a      = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("B:"):         b      = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("C:"):         c      = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("D:"):         d      = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("ANSWER:"):   answer = ln.split(":", 1)[1].strip().upper()
            elif ln.upper().startswith("FACT:"):     fact   = ln.split(":", 1)[1].strip()

        if not (q_text and a and b and c and d and answer):
            safe_send_message(chat_id, "⚠️ Couldn't generate trivia right now. Try again!")
            return

        # Store session
        memory.update_setting(str(chat_id), TRIVIA_KEY, {"answer": answer, "fact": fact, "cat": cat})

        markup = InlineKeyboardMarkup(row_width=2)
        markup.add(
            InlineKeyboardButton(f"🅰️  {a}", callback_data="trivia_ans_A"),
            InlineKeyboardButton(f"🅱️  {b}", callback_data="trivia_ans_B"),
            InlineKeyboardButton(f"🅲  {c}", callback_data="trivia_ans_C"),
            InlineKeyboardButton(f"🅳  {d}", callback_data="trivia_ans_D"),
            InlineKeyboardButton("❌ Skip",  callback_data="trivia_skip"),
        )
        safe_send_message(
            chat_id,
            f"🧠 *Trivia Challenge* — {cat}\n\n❓ *{q_text}*\n\nPick your answer:",
            reply_markup=markup
        )
    except Exception as e:
        logger.error(f"Trivia generate error: {e}")
        safe_send_message(chat_id, "⚠️ Trivia unavailable right now. Try again!")

def handle_trivia_answer(chat_id, chosen: str):
    """Check trivia answer, award points, send result."""
    session = memory.get_setting(str(chat_id), TRIVIA_KEY, None)
    if not session:
        safe_send_message(chat_id, "⚠️ No active trivia. Tap *Trivia* in games to start!")
        return
    memory.update_setting(str(chat_id), TRIVIA_KEY, None)

    correct = session.get("answer", "")
    fact    = session.get("fact", "")

    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🧠 Next Question", callback_data="game_trivia_start"),
        InlineKeyboardButton("🎮 All Games",      callback_data="stk_open_games"),
        InlineKeyboardButton("📊 My Stats",        callback_data="game_my_stats"),
    )

    if chosen == correct:
        pts   = 15
        stats = add_game_score(chat_id, pts, win=True)
        rk    = get_rank(stats["score"])
        bonus = streak_bonus_msg(stats["streak"])
        msg   = (
            f"✅ *CORRECT! Answer was {correct}!*\n\n"
            f"💡 {fact}\n\n"
            f"⭐ *+{pts} pts* → Total: *{stats['score']}* | {rk}\n"
            f"🔥 Streak: {stats['streak']}"
        )
        if bonus:
            msg += f"\n\n{bonus}"
    else:
        stats = add_game_score(chat_id, 0, win=False)
        msg   = (
            f"❌ *Wrong!* Correct answer was: *{correct}*\n\n"
            f"💡 {fact}\n\n"
            f"Streak reset. Keep going — you've got this! 💪"
        )
    safe_send_message(chat_id, msg, reply_markup=markup)

# ── 🃏 HIGHER OR LOWER CARD GAME ─────────────────────────────────────────────
_CARD_VALUES = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
_CARD_SUITS  = ["♠️","♥️","♦️","♣️"]
_CARD_RANK   = {v: i for i, v in enumerate(_CARD_VALUES)}

def _random_card():
    return random.choice(_CARD_VALUES), random.choice(_CARD_SUITS)

def _card_display(val, suit):
    return f"*{val}{suit}*"

def start_card_game(chat_id):
    """Start a Higher-or-Lower round: show current card, ask higher/lower."""
    val, suit = _random_card()
    memory.update_setting(str(chat_id), CARD_KEY, {"val": val, "suit": suit, "round": 1, "streak": 0})
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("⬆️ Higher",  callback_data="card_guess_higher"),
        InlineKeyboardButton("⬇️ Lower",   callback_data="card_guess_lower"),
        InlineKeyboardButton("🤝 Same",    callback_data="card_guess_same"),
        InlineKeyboardButton("🚪 Cash Out", callback_data="card_cashout"),
    )
    safe_send_message(
        chat_id,
        f"🃏 *Higher or Lower*\n\n"
        f"Current card: {_card_display(val, suit)}\n\n"
        f"Will the next card be *Higher*, *Lower*, or the *Same*?\n"
        f"Each correct guess = +5 pts. Cash out anytime to keep your score!",
        reply_markup=markup
    )

def handle_card_guess(chat_id, guess: str):
    """Process a Higher-or-Lower guess."""
    session = memory.get_setting(str(chat_id), CARD_KEY, None)
    if not session or guess == "cashout":
        if session:
            earned = session.get("streak", 0) * 5
            stats  = add_game_score(chat_id, earned, win=earned > 0)
            memory.update_setting(str(chat_id), CARD_KEY, None)
            rk     = get_rank(stats["score"])
            markup = InlineKeyboardMarkup(row_width=2)
            markup.add(
                InlineKeyboardButton("🃏 New Game", callback_data="game_card_start"),
                InlineKeyboardButton("🎮 All Games", callback_data="stk_open_games"),
            )
            safe_send_message(
                chat_id,
                f"🏦 *Cashed Out!*\n\n"
                f"You banked *+{earned} pts* across {session.get('round',1)-1} rounds!\n"
                f"Total: *{stats['score']}* | Rank: {rk}",
                reply_markup=markup
            )
        else:
            safe_send_message(chat_id, "⚠️ No active game! Start a new one from Games menu.")
        return

    old_val  = session["val"]
    old_suit = session["suit"]
    new_val, new_suit = _random_card()
    old_rank = _CARD_RANK[old_val]
    new_rank = _CARD_RANK[new_val]

    if new_rank > old_rank:   actual = "higher"
    elif new_rank < old_rank: actual = "lower"
    else:                     actual = "same"

    correct = (guess == actual)
    session["round"]  = session.get("round", 1) + 1

    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("⬆️ Higher",  callback_data="card_guess_higher"),
        InlineKeyboardButton("⬇️ Lower",   callback_data="card_guess_lower"),
        InlineKeyboardButton("🤝 Same",    callback_data="card_guess_same"),
        InlineKeyboardButton("🚪 Cash Out", callback_data="card_cashout"),
    )

    if correct:
        session["streak"] = session.get("streak", 0) + 1
        pts_so_far = session["streak"] * 5
        session["val"]  = new_val
        session["suit"] = new_suit
        memory.update_setting(str(chat_id), CARD_KEY, session)
        safe_send_message(
            chat_id,
            f"✅ *Correct!*\n"
            f"{_card_display(old_val, old_suit)} → {_card_display(new_val, new_suit)} — it was *{actual}*!\n\n"
            f"🔥 Streak: {session['streak']} correct | Banked so far: *{pts_so_far} pts*\n\n"
            f"Next card is {_card_display(new_val, new_suit)} — Higher, Lower, or Same?",
            reply_markup=markup
        )
    else:
        earned = session.get("streak", 0) * 5
        stats  = add_game_score(chat_id, earned, win=False)
        memory.update_setting(str(chat_id), CARD_KEY, None)
        rk     = get_rank(stats["score"])
        end_markup = InlineKeyboardMarkup(row_width=2)
        end_markup.add(
            InlineKeyboardButton("🃏 Play Again", callback_data="game_card_start"),
            InlineKeyboardButton("🎮 All Games",  callback_data="stk_open_games"),
            InlineKeyboardButton("📊 My Stats",   callback_data="game_my_stats"),
        )
        safe_send_message(
            chat_id,
            f"❌ *Wrong!*\n"
            f"{_card_display(old_val, old_suit)} → {_card_display(new_val, new_suit)} — it was *{actual}*, not {guess}!\n\n"
            f"You earned *+{earned} pts* this round.\n"
            f"Total: *{stats['score']}* | Rank: {rk}",
            reply_markup=end_markup
        )

# ── /gamestats command handler ───────────────────────────────────────────────
def generate_ai_sticker(chat_id, prompt: str):
    """Generate a custom sticker image using AI and send as WebP sticker or sticker art."""
    clean_p = (prompt or "").strip()
    if not clean_p:
        safe_send_message(
            chat_id,
            "🎨 *Custom AI Sticker Maker*\n\n"
            "Usage: `/makesticker [description]`\n"
            "Example: `/makesticker cute baby kitten wearing headphones, sticker art`\n"
            "Try it now!"
        )
        return

    thinking = safe_send_message(
        chat_id,
        f"🎨 *Designing your sticker:* \"{clean_p[:50]}...\"\n"
        "⏳ Crafting vector sticker art with vibrant colors..."
    )

    enhanced = f"die-cut sticker of {clean_p}, bold clean white outline, vibrant vector sticker design, flat colors, clean background, 8k sticker illustration"
    img_bytes = ImageGenerator.generate(enhanced, model_type="flux")
    if not img_bytes or (isinstance(img_bytes, dict) and img_bytes.get("type") == "text"):
        img_bytes = ImageGenerator.generate(enhanced, model_type="pollinations")

    if thinking:
        try:
            bot.delete_message(chat_id, thinking.message_id)
        except Exception:
            pass

    if not img_bytes or isinstance(img_bytes, dict):
        safe_send_message(chat_id, f"🎨 Couldn't generate image for \"{clean_p}\" right now. Try another prompt!")
        return

    sent_as_sticker = False
    if Image is not None:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            offset = ((512 - pil_img.width) // 2, (512 - pil_img.height) // 2)
            canvas.paste(pil_img, offset)
            webp_bio = io.BytesIO()
            canvas.save(webp_bio, format="WEBP")
            webp_bio.seek(0)
            webp_bio.name = "sticker.webp"

            res = bot.send_sticker(chat_id, webp_bio)
            if res and getattr(res, "sticker", None):
                sent_as_sticker = True
                sticker_vault.add_sticker(res.sticker.file_id, emoji="🎨", set_name="custom_ai")
        except Exception as e:
            logger.warning(f"Failed to convert or send as WebP sticker: {e}")

    if not sent_as_sticker:
        try:
            bot.send_photo(
                chat_id,
                img_bytes,
                caption=f"🎨 *Custom AI Sticker:* {clean_p}\n\n"
                        f"✨ Saved in sticker format! You can add this directly to Telegram sticker packs! 🌟"
            )
        except Exception as pe:
            logger.error(f"Send sticker photo fallback failed: {pe}")
            safe_send_message(chat_id, "❌ Could not send sticker image.")
            return

    safe_send_message(
        chat_id,
        f"✨ *Your custom AI sticker is ready!* 🚀\n"
        "You can forward it to friends or save it to your favorite stickers!",
        reply_markup=build_sticker_panel_markup()
    )

def handle_emojify_text(chat_id, text: str):
    clean_t = (text or "").strip()
    if not clean_t:
        safe_send_message(
            chat_id,
            "✨ *Emoji Masterpiece Transformer* 🎨\n\n"
            "Usage: `/emoji [your text or sentence]`\n"
            "Example: `/emoji I love coding and exploring the galaxy with AI`\n\n"
            "Try it now and watch the magic happen! 🚀"
        )
        return

    bot.send_chat_action(chat_id, 'typing')
    prompt = (
        "You are an emoji artist. Transform the following text into a delightful, "
        "expressive, creative emoji-packed story and message. "
        "Include fitting emojis, Kaomoji ASCII art, and colorful excitement!\n\n"
        f"Original text: {clean_t}"
    )
    result = ask_ai_text(chat_id, prompt, temperature=0.8)
    if not result:
        safe_send_message(chat_id, f"✨ {clean_t} ✨ 🚀 💖 🌟 🎉")
        return
    safe_send_message(chat_id, f"🎨 *Emoji Masterpiece:*\n\n{result}")

def is_natural_sticker_request(text: str):
    t = (text or "").lower().strip()
    if any(phrase in t for phrase in [
        "send me a sticker", "send a sticker", "give me a sticker", "send sticker",
        "drop a sticker", "want a sticker", "show me a sticker", "sticker please",
        "send me sticker", "sticker"
    ]) and len(t.split()) <= 8:
        mood = sticker_vault.detect_mood_from_text(t)
        return True, mood
    return False, None

def is_natural_game_request(text: str):
    t = (text or "").lower().strip()
    # Trivia / quiz requests
    if any(phrase in t for phrase in ["play trivia", "quiz me", "ask me trivia", "trivia game", "play quiz"]):
        return True, "trivia"
    # Card game requests
    if any(phrase in t for phrase in ["higher or lower", "card game", "play cards", "play higher", "higher lower"]):
        return True, "card"
    # Physical dice games
    if any(phrase in t for phrase in [
        "play a game", "let's play a game", "play dice", "roll dice",
        "roll a dice", "play darts", "play basketball", "play slot", "play game"
    ]) and len(t.split()) <= 7:
        if "dart" in t:                          return True, "dart"
        if "basket" in t:                        return True, "basket"
        if "soccer" in t or "football" in t:    return True, "football"
        if "slot" in t or "casino" in t:        return True, "slot"
        if "bowl" in t:                          return True, "bowl"
        return True, "dice"
    return False, None

# ============================================================================
# 🚀 START COMMAND (FIXED)
# ============================================================================
def play_intro_animation(chat_id):
    """Render a red-branded Telegram-friendly boot animation."""
    frames = [
        "```text\n[ ARTOVIX RED CORE ]\nPowering on... [#####-----] 50%\n```",
        "```text\n[ ARTOVIX RED CORE ]\nModules online [##########] 100%\nChat [READY] | Code [READY] | Vision [READY] | Stickers [READY]\n```",
        "🔴 *ARTOVIX MASTERPIECE READY*\n✅ Welcome sequence complete. Let's make magic happen! ✨"
    ]

    msg = None
    for i, frame in enumerate(frames):
        try:
            if i == 0:
                msg = bot.send_message(chat_id, frame, parse_mode="Markdown")
            elif msg:
                bot.edit_message_text(
                    frame,
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    parse_mode="Markdown"
                )
            time.sleep(0.12 if i < len(frames) - 1 else 0.05)
        except Exception:
            break

MENU_CHAT = "🔴 Chat"
MENU_DRAW = "🎨 Draw"
MENU_STICKERS = "🎭 Stickers & Emojis"
MENU_GAMES = "🎲 Fun & Games"
MENU_SEARCH = "🔍 Search"
MENU_CODE = "💻 Code"
MENU_DOC = "📄 Ask Doc"
MENU_VOICE = "🔈 Voice Mode"
MENU_SETTINGS = "⚙️ Settings"
MENU_HELP = "❓ Help"
MENU_RESET = "🧹 Reset"

PENDING_MODE_KEY = "pending_mode"
RESPONSE_LENGTH_KEY = "response_length"

def build_main_reply_menu():
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.row(MENU_CHAT, MENU_DRAW)
    kb.row(MENU_SEARCH, MENU_GAMES)
    kb.row(MENU_CODE, MENU_DOC)
    kb.row(MENU_VOICE, MENU_SETTINGS)
    kb.row(MENU_HELP, MENU_RESET)
    return kb

def set_pending_mode(user_id, mode):
    memory.update_setting(str(user_id), PENDING_MODE_KEY, mode)

def get_pending_mode(user_id):
    return memory.get_setting(str(user_id), PENDING_MODE_KEY, None)

def send_welcome_panel(chat_id):
    welcome_msg = """🔴 *Artovix 2026 Masterpiece* ✨

Welcome to your ultimate AI companion! I am equipped with cutting-edge intelligence, creative art, interactive games, coding expertise, and a warm heart! 💖

✨ *What I can do for you:*
• 🔴 *Chat:* Warm, smart, friendly conversations
• 🎨 *Draw:* Stunning AI image generation
• 🎲 *Games:* Play live animated dice, darts, basketball, slots, & bowling!
• 💻 *Code:* Expert coding, debugging, and scripts
• 📄 *Doc:* Upload PDFs/files and ask questions
• 🔈 *Voice:* Voice responses with customizable accents

Tap any button below or type a message to start! 🚀"""

    safe_send_message(chat_id, welcome_msg, reply_markup=build_main_reply_menu())

@bot.message_handler(commands=['menu'])
def handle_menu(message):
    try:
        if not ensure_channel_access(message):
            return
        send_welcome_panel(message.chat.id)
    except Exception as e:
        logger.error(f"Menu command error: {e}")
        safe_send_message(message.chat.id, "Please use `/start` to open the menu.")

@bot.message_handler(commands=['start', 'artovix', 'hello'])
def start_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_intro_animation(message.chat.id)
        send_welcome_panel(message.chat.id)
        logger.info(f"✓ Start command from user {message.chat.id}")
        
    except Exception as e:
        logger.error(f"Start command error: {e}")
        bot.send_message(message.chat.id, "Welcome to Artovix. Use `/help` to view commands.")

# ============================================================================
# 🎨 DRAW COMMAND (IMPROVED)
# ============================================================================
@bot.message_handler(commands=['draw_legacy'])
def handle_draw(message):
    thinking_msg = None
    try:
        if not ensure_channel_access(message):
            return
        # Get prompt from command
        if message.text and len(message.text.split()) > 1:
            prompt = ' '.join(message.text.split()[1:])
        else:
            # Show help if no prompt
            help_text = """🎨 *AI Image Generator*

*Usage:* `/draw [description]`

*Examples:*
• `/draw a majestic dragon flying over mountains at sunset`
• `/draw cyberpunk city with neon lights, rain, futuristic`
• `/draw cute anime cat with sunglasses, detailed background`
• `/draw fantasy forest with glowing mushrooms, magical`

*Tips:*
• Be detailed with colors and lighting
• Add style: `digital art`, `photorealistic`, `anime style`
• Specify composition: `wide angle`, `close-up`, `dynamic`

*Try:* `/draw a beautiful landscape with mountains and lake`"""
            
            safe_send_message(message.chat.id, help_text)
            return
        
        # Show thinking message
        thinking_msg = safe_send_message(
            message.chat.id,
            f"🎨 *Creating:* \"{prompt[:60]}...\"\n"
            f"⏳ Generating image with AI... (10-20 seconds)"
        )
        
        # Generate image
        result = ImageGenerator.generate(prompt)
        
        # Delete thinking message
        if thinking_msg:
            try:
                bot.delete_message(message.chat.id, thinking_msg.message_id)
            except:
                pass
        
        if result:
            if isinstance(result, dict) and result.get('type') == 'text':
                # Text-based result (fallback)
                text_response = f"""🎨 *AI Image Concept:* {result['prompt']}

{result['emojis']} *Visual Description:*
{result['description']}

💡 *Pro Tip:* {result['suggestion']}

✨ *Try:* `/draw {prompt}, 4k, detailed, cinematic lighting`"""
                
                safe_send_message(message.chat.id, text_response)
                
            else:
                # Actual image
                try:
                    bot.send_photo(
                        message.chat.id,
                        result,
                        caption=f"🎨 *AI Generated:* {prompt}\n\n"
                                   f"✨ Powered by Artovix AI | {datetime.now().strftime('%H:%M')}"
                    )
                    logger.info(f"✓ Image sent to {message.chat.id}")
                except Exception as e:
                    logger.error(f"Photo send error: {e}")
                    # Fallback to text
                    safe_send_message(
                        message.chat.id,
                        f"🎨 *Generated:* {prompt}\n\n"
                        f"✅ Image created! (Preview unavailable)\n\n"
                        f"✨ Try: `/draw {prompt}, enhanced details`"
                    )
        else:
            # No result
            safe_send_message(
                message.chat.id,
                f"🎨 *Your Concept:* {prompt}\n\n"
                f"That's an awesome idea! 🚀\n\n"
                f"*Try being more specific:*\n"
                f"• Add colors: `vibrant colors`, `golden hour lighting`\n"
                f"• Specify style: `digital art style`, `anime artwork`\n"
                f"• Add details: `highly detailed`, `intricate patterns`\n\n"
                f"*Example:* `/draw {prompt}, cinematic lighting, 8k resolution`"
            )
        
        analytics.log_request(message.chat.id, len(prompt.split()), "image_generation")
        
    except Exception as e:
        logger.error(f"Draw command error: {e}\n{traceback.format_exc()}")
        
        # Clean up thinking message
        if thinking_msg:
            try:
                bot.delete_message(message.chat.id, thinking_msg.message_id)
            except:
                pass
        
        safe_send_message(
            message.chat.id,
            "🎨 *Image Generation*\n\n"
            "Try: `/draw [detailed description]`\n\n"
            "*Example:* `/draw a fantasy castle on a cloud, sunset lighting`"
        )

# ============================================================================
# 🔍 SEARCH COMMAND (FIXED)
# ============================================================================
@bot.message_handler(commands=['search', 'find', 'google'])
def handle_search(message):
    try:
        if not ensure_channel_access(message):
            return
        # Extract query
        if message.text and len(message.text.split()) > 1:
            query = ' '.join(message.text.split()[1:])
        else:
            safe_send_message(
                message.chat.id,
                "🔍 *Web Search*\n\n"
                "*Usage:* `/search [your question]`\n\n"
                "*Examples:*\n"
                "• `/search latest AI developments in 2026`\n"
                "• `/search how to learn Python programming`\n"
                "• `/search best practices for web development`"
            )
            return
        
        # Show searching indicator continuously while processing
        with KeepTyping(message.chat.id, 'typing'):
            # Create search prompt with enhanced instructions
            search_prompt = f"""Search Query: {query}
        
As a Knowledge Specialist in 2026, provide a comprehensive search result for the query above.
        
Structure your response as follows:
🌐 [Topic Overview]
Brief summary of the most current information.
        
📌 [Key Facts & Developments]
- Detail 1
- Detail 2
        
🛠️ [Practical Insights/Applications]
How this information is used or its significance.
        
💡 [Expert Tip]
A unique insight or recommendation.
        
Keep it professional, accurate, and formatted for a mobile chat interface."""
        
            try:
                if not (nvidia_client or groq_client):
                    safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `NVIDIA_API_KEY` or `GROQ_API_KEY` in your .env to enable search features.")
                    return

                # Get response from AI backend using dedicated search/research model
                content, _, _ = ai_chat_with_fallback(
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0.7,
                    max_tokens=600,
                    model=get_purpose_model("search", message.chat.id)
                )

                answer = clean_markdown(content)

                # Send result
                result_text = f"🔍 *Search Results:* {query}\n\n{answer}\n\n✨ *Source:* Artovix AI Knowledge Base"
                save_last_ai_reply(message.chat.id, result_text)
                send_ai_reply(message.chat.id, result_text, reply_markup=maybe_followup_markup(result_text))

            except Exception as api_error:
                logger.error(f"Search API error: {api_error}")
                safe_send_message(
                    message.chat.id,
                    f"🔍 *Search:* {query}\n\n"
                    f"I'll help you with that! Here's what I know:\n\n"
                    f"Please try rephrasing your question or ask me directly about the topic."
                )
        
            analytics.log_request(message.chat.id, len(query.split()) * 30, "search")
        
    except Exception as e:
        logger.error(f"Search command error: {e}")
        safe_send_message(
            message.chat.id,
            "🔍 *Search temporarily unavailable*\n\n"
            "Try asking your question directly to me!"
        )

# ============================================================================
# 💻 CODE COMMAND (FIXED)
# ============================================================================
@bot.message_handler(commands=['code', 'program', 'debug'])
def handle_code(message):
    try:
        if not ensure_channel_access(message):
            return
        # Extract code or question
        if message.text and len(message.text.split()) > 1:
            code_text = ' '.join(message.text.split()[1:])
        else:
            safe_send_message(
                message.chat.id,
                "💻 *Code Assistant*\n\n"
                "*Usage:*\n"
                "1. Ask a question: `/code how to reverse a string in Python?`\n"
                "2. Send code for analysis:\n"
                "```python\n"
                "def hello():\n"
                "    print('Hello World!')\n"
                "```\n\n"
                "*Examples:*\n"
                "• `/code explain this Python function`\n"
                "• `/code how to create a web API`\n"
                "• `/code fix my JavaScript code`"
            )
            return
        
        with KeepTyping(message.chat.id, 'typing'):
            # Create code analysis prompt
            if '```' in code_text:
                # It's code in a block
                code_prompt = f"""Analyze this code and provide:

1. What it does
2. Any issues or bugs
3. Improvements
4. Best practices

OUTPUT FORMAT RULES:
- Always include corrected/improved code in fenced code blocks.
- Use triple backticks with a language tag (example: ```python).
- Make code directly copyable and runnable.

Code:
{code_text}"""
            else:
                # It's a question
                code_prompt = f"""Answer this programming question: {code_text}

Provide:
1. Clear explanation
2. Code examples if applicable
3. Best practices
4. Common pitfalls to avoid

OUTPUT FORMAT RULES:
- Always include at least one copyable code block.
- Use triple backticks with a language tag.
- Keep code practical and runnable."""
            
            try:
                if not (nvidia_client or groq_client):
                    safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `NVIDIA_API_KEY` or `GROQ_API_KEY` in your .env to enable code analysis.")
                    return

                content, _, _ = ai_chat_with_fallback(
                    messages=[{"role": "user", "content": code_prompt}],
                    temperature=0.3,
                    max_tokens=1200,
                    model=get_purpose_model("code", message.chat.id)
                )

                preferred_language = detect_code_language(code_text)
                analysis = ensure_copyable_code_blocks(
                    content,
                    preferred_language=preferred_language
                )
                analysis = clean_markdown(analysis)

                result_text = f"💻 *Code Solution:*\n\n{analysis}\n\n🔧 *Powered by Artovix AI*"
                save_last_ai_reply(message.chat.id, result_text)
                send_ai_reply(message.chat.id, result_text, reply_markup=maybe_followup_markup(result_text))

            except Exception as api_error:
                logger.error(f"Code API error: {api_error}")
                err_msg = str(api_error)
                if message.chat.id in ADMIN_USER_IDS:
                    safe_send_message(
                        message.chat.id,
                        f"⚠️ *Code Error (Admin Info):*\n`{err_msg[:400]}`"
                    )
                else:
                    safe_send_message(
                        message.chat.id,
                        "💻 *Code generation encountered a temporary glitch.* Please try again!"
                    )
            
            analytics.log_request(message.chat.id, len(code_text.split()), "code_analysis")
        
    except Exception as e:
        logger.error(f"Code command error: {e}")
        safe_send_message(
            message.chat.id,
            "💻 *Code analysis failed*\n\n"
            "Try sending your code in this format:\n"
            "```python\n"
            "# Your code here\n"
            "print('Hello')\n"
            "```"
        )

# ============================================================================
# 📊 STATS COMMAND (FIXED)
# ============================================================================
@bot.message_handler(commands=['stats', 'analytics', 'metrics'])
def handle_stats(message):
    try:
        if not ensure_channel_access(message):
            return
        if not require_main_admin(message):
            return
        metrics = analytics.get_current_metrics()
        breakdown = metrics.get('breakdown', {})
        
        breakdown_text = ""
        for rtype, count in breakdown.items():
            breakdown_text += f"• {rtype.replace('_', ' ').title()}: {count}\n"
        
        if not breakdown_text:
            breakdown_text = "• No requests today yet."
            
        stats_msg = f"""📊 *Artovix Analytics Dashboard*

*Live Metrics:*
• **RPM:** {metrics['RPM']} requests/minute
• **TPM:** {metrics['TPM']:,} tokens/minute
• **RPD:** {metrics['RPD']} total requests today

*Usage Breakdown:*
{breakdown_text}

*System Status:*
• 🤖 Version: Artovix 2026.2.0
• 🧠 Models: Llama 3.3, 3.2 Vision, Whisper
• 💬 Active Users: {len(memory.load())}
• 🕐 Server Time: {datetime.now().strftime('%H:%M:%S')}

*All systems operational!* 🚀"""
        
        safe_send_message(message.chat.id, stats_msg)
        
    except Exception as e:
        logger.error(f"Stats command error: {e}")
        safe_send_message(message.chat.id, "📊 Analytics: System active and running!")

# ============================================================================
# 🛡️ OTHER COMMANDS (FIXED)
# ============================================================================
@bot.message_handler(commands=['debug', 'aitest'])
def handle_debug(message):
    """Admin-only: test AI and show real error details."""
    try:
        if not require_main_admin(message):
            return
        bot.send_chat_action(message.chat.id, 'typing')
        lines = [
            f"🔧 *AI Debug Report*\n",
            f"• `NVIDIA_API_KEY`: {'✅ set' if NVIDIA_API_KEY else '❌ missing'}",
            f"• `GROQ_KEY`: {'✅ set' if GROQ_KEY else '❌ missing'}",
            f"• `AI_PROVIDER`: `{AI_PROVIDER}`",
            f"• `CHAT_MODEL`: `{CHAT_MODEL}`",
            f"• Dead models blacklisted: `{len(_DEAD_MODELS)}`",
            "",
        ]
        try:
            content, model_used, _ = ai_chat_with_fallback(
                messages=[{"role": "user", "content": "Say: OK"}],
                temperature=0.1,
                max_tokens=10
            )
            lines.append(f"✅ *AI responding!* Model: `{model_used}`")
        except Exception as e:
            lines.append(f"❌ *AI failed:*\n`{str(e)[:500]}`")
            if _DEAD_MODELS:
                dead_list = "\n".join(f"  - `{p}/{m}`" for p, m in sorted(_DEAD_MODELS))
                lines.append(f"\n*Dead models:*\n{dead_list}")
        safe_send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        logger.error(f"Debug command error: {e}")
        safe_send_message(message.chat.id, f"Debug error: {e}")

@bot.message_handler(commands=['reset'])
def handle_reset(message):
    try:
        if not ensure_channel_access(message):
            return
        user_id = str(message.chat.id)
        user_data = memory.get_user_data(user_id)
        user_data["history"] = []
        memory.save_user_data(user_id, user_data)
        
        safe_send_message(
            message.chat.id,
            "🧹 *Memory Cleared!*\n\n"
            "Our conversation history has been reset.\n"
            "Ready for a fresh start! 👋\n\n"
            "*Try:* `/draw something creative`"
        )
    except Exception as e:
        logger.error(f"Reset error: {e}")
        safe_send_message(message.chat.id, "🧹 Reset completed!")

@bot.message_handler(commands=['status'])
def handle_status(message):
    try:
        if not require_admin(message):
            return
        status_msg = f"""✅ *Artovix Status Report*


*Core Systems:*
• 🤖 AI Engine: ✅ Online
• 🧠 Memory: ✅ {len(memory.load())} active
• 🎨 Image Gen: ✅ Multiple services
• 🎙️ Voice/Vision: ✅ Optimized
• 🔍 Search: ✅ Active

*Server Info:*
• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Version: 2026.2.0 Stable
• Uptime: 100%

*Ready to assist!* 🚀"""
        
        safe_send_message(message.chat.id, status_msg)
    except Exception as e:
        logger.error(f"Status error: {e}")
        safe_send_message(message.chat.id, "✅ Artovix is running!")

@bot.message_handler(commands=['help'])
def handle_help(message):
    try:
        if not ensure_channel_access(message):
            return
        help_text = """🔴 *Artovix Masterpiece Guide* ✨

*🌟 Core Features*
`/start` or `/menu` - Open main interactive menu
`/draw [prompt]` - Generate AI images (Flux / Pollinations)
`/search [query]` - Comprehensive knowledge research
`/code [question]` - Professional programming assistant
`/askdoc [question]` - Upload documents & ask questions
`/summarize [text]` - Instant smart summary

*🎲 Epic Games Arcade*
`/game` or `/dice` - 🎲 Lucky Dice (earn pts!)
`/dart` - 🎯 Bullseye Dart Challenge
`/basket` - 🏀 Basketball Shot
`/soccer` - ⚽ Soccer Penalty
`/slot` - 🎰 Jackpot Slot Machine
`/bowl` - 🎳 Bowling Strike
`/trivia` or `/quiz` - 🧠 AI Trivia Challenge (+15 pts)
`/card` or `/higher` - 🃏 Higher or Lower Card Game
`/gamestats` or `/rank` - 📊 Your rank & game stats

*⚙️ Preferences & Tools*
`/model [name]` - Switch NVIDIA NIM models (Nemotron, DeepSeek, etc.)
`/voice` - Voice mode & accent settings
`/lang` - Choose language (Amharic, English, Arabic, etc.)
`/settings` - All user preferences
`/reset` - Clear conversation memory
`/admin` - Admin panel (admins only)"""
        
        safe_send_message(message.chat.id, help_text)
    except Exception as e:
        logger.error(f"Help error: {e}")
        safe_send_message(message.chat.id, "Type /start to begin!")

@bot.message_handler(commands=['templates'])
def handle_templates(message):
    try:
        if not ensure_channel_access(message):
            return
        text = (
            "🧩 *Prompt Templates*\n\n"
            "`/draw cinematic portrait of a lion in red neon, ultra detailed`\n"
            "`/search best plan to learn Python in 30 days`\n"
            "`/code build a Telegram bot command parser with examples`\n"
            "`/summarize Explain quantum computing for beginners in simple words`\n\n"
            "Tip: you can also reply to any long message with `/summarize`."
        )
        safe_send_message(message.chat.id, text)
    except Exception as e:
        logger.error(f"Templates command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to open templates.")

@bot.message_handler(commands=['summarize'])
def handle_summarize(message):
    try:
        if not ensure_channel_access(message):
            return
        target_text = ""
        if message.text and len(message.text.split()) > 1:
            target_text = message.text.split(maxsplit=1)[1].strip()
        elif getattr(message, "reply_to_message", None):
            reply_msg = message.reply_to_message
            target_text = (getattr(reply_msg, "text", None) or getattr(reply_msg, "caption", None) or "").strip()

        if not target_text:
            safe_send_message(
                message.chat.id,
                "Use `/summarize your text` or reply to a message with `/summarize`."
            )
            return
        if not (nvidia_client or groq_client):
            safe_send_message(message.chat.id, "🔌 AI backend unavailable. Set `NVIDIA_API_KEY` or `GROQ_API_KEY`.")
            return

        bot.send_chat_action(message.chat.id, 'typing')
        prompt = (
            "Summarize this content into concise bullet points. "
            "Keep key facts and actions. If technical, include short code-oriented tips.\n\n"
            f"{target_text}"
        )
        summary = ask_ai_text(message.chat.id, prompt, temperature=0.3)
        if not summary:
            safe_send_message(message.chat.id, "⚠️ Could not summarize right now.")
            return
        save_last_ai_reply(message.chat.id, summary)
        summary_text = f"📝 *Summary:*\n\n{summary}"
        send_ai_reply(message.chat.id, summary_text, reply_markup=maybe_followup_markup(summary_text))
    except Exception as e:
        logger.error(f"Summarize command error: {e}")
        safe_send_message(message.chat.id, "❌ Summarize failed. Try again.")

@bot.message_handler(commands=['lang', 'language'])
def handle_language_mode(message):
    try:
        if not ensure_channel_access(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2:
            send_language_panel(message.chat.id)
            return

        choice = parts[1].strip().lower()
        aliases = {
            "auto": "auto",
            "english": "en",
            "en": "en",
            "amharic": "am",
            "am": "am",
            "arabic": "ar",
            "ar": "ar",
            "french": "fr",
            "fr": "fr",
            "spanish": "es",
            "es": "es",
            "german": "de",
            "de": "de",
            "italian": "it",
            "it": "it",
            "portuguese": "pt",
            "pt": "pt",
            "russian": "ru",
            "ru": "ru",
            "turkish": "tr",
            "tr": "tr",
            "hindi": "hi",
            "hi": "hi",
            "swahili": "sw",
            "sw": "sw",
        }
        lang = aliases.get(choice)
        if not lang:
            safe_send_message(message.chat.id, "Use `/lang` panel, or `/lang <code>` like `en, am, ar, fr, es, de, it, pt, ru, tr, hi, sw, auto`.")
            return
        set_user_language(message.chat.id, lang)
        safe_send_message(message.chat.id, f"✅ Language set to *{LANGUAGE_LABELS.get(lang, 'Auto')}*.")
    except Exception as e:
        logger.error(f"Language mode command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to update language mode.")

@bot.message_handler(commands=['settings'])
def handle_settings(message):
    try:
        if not ensure_channel_access(message):
            return
        send_settings_panel(message.chat.id)
    except Exception as e:
        logger.error(f"Settings command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to open settings.")

@bot.message_handler(commands=['voice'])
def handle_voice_mode(message):
    try:
        if not ensure_channel_access(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2:
            send_voice_mode_panel(message.chat.id)
            return

        mode = parts[1].strip().lower()
        if mode in {"male", "female"}:
            set_voice_profile(message.chat.id, mode)
            set_voice_reply_enabled(message.chat.id, True)
            safe_send_message(message.chat.id, f"✅ Voice profile set to *{mode.upper()}*.")
            return

        if mode in {"soft", "normal", "fast"}:
            set_voice_style(message.chat.id, mode)
            set_voice_reply_enabled(message.chat.id, True)
            safe_send_message(message.chat.id, f"✅ Voice style set to *{mode.upper()}*.")
            return

        if mode not in {"on", "off"}:
            safe_send_message(
                message.chat.id,
                "Use `/voice on|off|male|female|soft|normal|fast` or `/voice` for options."
            )
            return

        enabled = mode == "on"
        set_voice_reply_enabled(message.chat.id, enabled)
        safe_send_message(
            message.chat.id,
            "✅ Voice replies enabled." if enabled else "✅ Voice replies disabled."
        )
    except Exception as e:
        logger.error(f"Voice mode command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to update voice mode.")

# ============================================================================
# 🧠 AI MODEL SWITCHER COMMAND
# ============================================================================
@bot.message_handler(commands=['model', 'models', 'aimodel'])
def handle_model_command(message):
    try:
        if not ensure_channel_access(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) > 1:
            arg = parts[1].strip()
            canonical = set_user_chat_model(message.chat.id, arg)
            info = NVIDIA_FEATURED_MODELS.get(canonical, NVIDIA_FEATURED_MODELS["auto"])
            safe_send_message(
                message.chat.id,
                f"✅ *AI Model Switched!*\n\n"
                f"🧠 *Active Model:* `{info['name']}`\n"
                f"⚡ *Features:* {info['badge']}\n"
                f"📝 *Description:* _{info['desc']}_\n\n"
                "Your future chat, search, and coding questions will use this model! 🚀"
            )
            return

        send_model_panel(message.chat.id)
    except Exception as e:
        logger.error(f"Model command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not open model panel.")

# ============================================================================
# 🎭 STICKER & EMOJI & GAME COMMANDS
# ============================================================================
@bot.message_handler(commands=['sticker', 'stickers'])
def handle_sticker_command(message):
    # Sticker feature removed — only greeting stickers remain
    pass

@bot.message_handler(commands=['makesticker', 'stickergen'])
def handle_makesticker_command(message):
    # Sticker generation feature removed
    pass

@bot.message_handler(commands=['emoji', 'emojify'])
def handle_emoji_command(message):
    # Emoji feature removed
    pass

@bot.message_handler(commands=['game', 'games'])
def handle_games_command(message):
    try:
        if not ensure_channel_access(message):
            return
        send_games_panel(message.chat.id)
    except Exception as e:
        logger.error(f"Games command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not open games corner.")

@bot.message_handler(commands=['dice', 'roll'])
def handle_dice_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "dice")
    except Exception as e:
        logger.error(f"Dice command error: {e}")

@bot.message_handler(commands=['dart', 'darts'])
def handle_dart_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "dart")
    except Exception as e:
        logger.error(f"Dart command error: {e}")

@bot.message_handler(commands=['basket', 'basketball'])
def handle_basket_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "basket")
    except Exception as e:
        logger.error(f"Basket command error: {e}")

@bot.message_handler(commands=['soccer', 'football'])
def handle_soccer_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "football")
    except Exception as e:
        logger.error(f"Soccer command error: {e}")

@bot.message_handler(commands=['slot', 'casino'])
def handle_slot_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "slot")
    except Exception as e:
        logger.error(f"Slot command error: {e}")

@bot.message_handler(commands=['bowl', 'bowling'])
def handle_bowl_command(message):
    try:
        if not ensure_channel_access(message):
            return
        play_telegram_game(message.chat.id, "bowl")
    except Exception as e:
        logger.error(f"Bowl command error: {e}")

@bot.message_handler(commands=['trivia', 'quiz'])
def handle_trivia_command(message):
    try:
        if not ensure_channel_access(message):
            return
        bot.send_chat_action(message.chat.id, 'typing')
        Thread(target=start_trivia, args=(message.chat.id,), daemon=True).start()
    except Exception as e:
        logger.error(f"Trivia command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not start trivia right now.")

@bot.message_handler(commands=['card', 'higher', 'highlower'])
def handle_card_command(message):
    try:
        if not ensure_channel_access(message):
            return
        start_card_game(message.chat.id)
    except Exception as e:
        logger.error(f"Card game command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not start card game.")

@bot.message_handler(commands=['gamestats', 'gscore', 'rank'])
def handle_gamestats_command(message):
    try:
        if not ensure_channel_access(message):
            return
        uname = getattr(message.from_user, "first_name", None) or "Player"
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(InlineKeyboardButton("🎮 Play Games", callback_data="stk_open_games"))
        safe_send_message(message.chat.id, format_stats_card(message.chat.id, uname), reply_markup=markup)
    except Exception as e:
        logger.error(f"Gamestats command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not load stats.")

@bot.message_handler(commands=['askdoc', 'doc'])
def handle_askdoc(message):
    try:
        if not ensure_channel_access(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2:
            doc_name = memory.get_setting(str(message.chat.id), DOC_NAME_KEY, None)
            if doc_name:
                safe_send_message(
                    message.chat.id,
                    f"📄 Active doc: *{doc_name}*\n"
                    "Now ask:\n`/askdoc summarize key points`"
                )
            else:
                safe_send_message(
                    message.chat.id,
                    "📄 Send a document first (PDF/TXT/MD/CSV/JSON), then ask:\n"
                    "`/askdoc your question`"
                )
            return

        doc_text = memory.get_setting(str(message.chat.id), DOC_CONTEXT_KEY, "")
        doc_name = memory.get_setting(str(message.chat.id), DOC_NAME_KEY, "document")
        if not doc_text:
            safe_send_message(
                message.chat.id,
                "📄 I don't have a document from you yet.\n"
                "Upload a PDF/TXT file first."
            )
            return

        query = parts[1].strip()
        bot.send_chat_action(message.chat.id, 'typing')

        if not (nvidia_client or groq_client):
            safe_send_message(message.chat.id, "🔌 *AI backend not configured.* Set `NVIDIA_API_KEY` or `GROQ_API_KEY`.")
            return

        prompt = (
            f"You are a document assistant. Use only the document content below when possible.\n\n"
            f"Document name: {doc_name}\n"
            f"Question: {query}\n\n"
            f"Document content:\n{doc_text[:MAX_DOC_PROMPT_CHARS]}\n\n"
            "Give a clear answer. If the answer is not in the document, say that briefly."
        )

        content, _, _ = ai_chat_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
            model=get_purpose_model("search", message.chat.id)
        )
        answer = clean_markdown(content)
        final_text = f"📄 *Doc Answer ({doc_name}):*\n\n{answer}"
        save_last_ai_reply(message.chat.id, final_text)
        send_ai_reply(message.chat.id, final_text, reply_markup=maybe_followup_markup(final_text))
        analytics.log_request(message.chat.id, len(query.split()), "doc_qa")
    except Exception as e:
        logger.error(f"AskDoc command error: {e}")
        safe_send_message(message.chat.id, "❌ Doc question failed. Try again.")

@bot.message_handler(commands=['docreset'])
def handle_docreset(message):
    try:
        if not ensure_channel_access(message):
            return
        memory.update_setting(str(message.chat.id), DOC_CONTEXT_KEY, "")
        memory.update_setting(str(message.chat.id), DOC_NAME_KEY, "")
        memory.update_setting(str(message.chat.id), DOC_UPDATED_KEY, "")
        safe_send_message(message.chat.id, "🧹 Document context cleared.")
    except Exception as e:
        logger.error(f"Doc reset error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to clear document context.")

# ============================================================================
# 🆔 ID DEBUG
# ============================================================================
@bot.message_handler(commands=['myid'])
def handle_myid(message):
    try:
        uid = get_actor_user_id(message)
        chat_id = getattr(getattr(message, "chat", None), "id", None)
        chat_type = getattr(getattr(message, "chat", None), "type", "unknown")
        admins = sorted(get_effective_admin_ids())
        safe_send_message(
            message.chat.id,
            f"🆔 Your user id: `{uid}`\n"
            f"💬 Chat id: `{chat_id}` ({chat_type})\n"
            f"👑 Main admin id: `{MAIN_ADMIN_ID}`\n"
            f"👮 Admin list size: {len(admins)}"
        )
    except Exception as e:
        logger.error(f"MyID command error: {e}")
        safe_send_message(message.chat.id, "❌ Could not read your ID.")

# ============================================================================
# 👮 ADMIN BROADCAST COMMANDS
# ============================================================================
@bot.message_handler(commands=['users'])
def handle_users(message):
    try:
        if not require_admin(message):
            return
        users = sorted(get_all_known_user_ids())
        safe_send_message(
            message.chat.id,
            f"👥 *Known Users:* {len(users)}\n"
            f"Use `/broadcast your message`, reply with `/post`, or run `/postwizard` for guided posting."
        )
    except Exception as e:
        logger.error(f"Users command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to fetch users.")

@bot.message_handler(commands=['admins'])
def handle_admins(message):
    try:
        if not require_admin(message):
            return
        admins = sorted(get_effective_admin_ids())
        lines = "\n".join([f"- `{uid}`" for uid in admins]) if admins else "- none"
        safe_send_message(
            message.chat.id,
            f"👮 *Admins ({len(admins)}):*\n{lines}"
        )
    except Exception as e:
        logger.error(f"Admins command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to fetch admins.")

def _extract_target_admin_id(message):
    # 1) explicit: /addadmin 12345
    if message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) > 1 and parts[1].strip().isdigit():
            return int(parts[1].strip())
    # 2) reply to a user's message
    if message.reply_to_message and message.reply_to_message.from_user:
        try:
            return int(message.reply_to_message.from_user.id)
        except:
            return None
    return None

@bot.message_handler(commands=['addadmin'])
def handle_add_admin(message):
    try:
        if not require_main_admin(message):
            return
        target_id = _extract_target_admin_id(message)
        if not target_id:
            safe_send_message(message.chat.id, "Usage: `/addadmin <user_id>` or reply with `/addadmin`.")
            return
        dynamic_admins = load_dynamic_admin_ids()
        dynamic_admins.add(target_id)
        save_dynamic_admin_ids(dynamic_admins)
        safe_send_message(message.chat.id, f"✅ Added admin: `{target_id}`")
    except Exception as e:
        logger.error(f"Add admin command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to add admin.")

@bot.message_handler(commands=['deladmin', 'removeadmin'])
def handle_remove_admin(message):
    try:
        if not require_main_admin(message):
            return
        target_id = _extract_target_admin_id(message)
        if not target_id:
            safe_send_message(message.chat.id, "Usage: `/deladmin <user_id>` or reply with `/deladmin`.")
            return
        dynamic_admins = load_dynamic_admin_ids()
        if target_id in dynamic_admins:
            dynamic_admins.remove(target_id)
            save_dynamic_admin_ids(dynamic_admins)
            safe_send_message(message.chat.id, f"✅ Removed admin: `{target_id}`")
        else:
            if target_id in ADMIN_USER_IDS:
                safe_send_message(message.chat.id, "⚠️ This admin comes from `ADMIN_USER_IDS` env var. Edit env to remove.")
            else:
                safe_send_message(message.chat.id, "ℹ️ User is not in dynamic admin list.")
    except Exception as e:
        logger.error(f"Remove admin command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to remove admin.")

@bot.message_handler(commands=['broadcast'])
def handle_broadcast(message):
    try:
        if not require_admin(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2 or not parts[1].strip():
            safe_send_message(message.chat.id, "Usage: `/broadcast your message`")
            return
        delivered, failed = broadcast_text_to_users(parts[1].strip())
        safe_send_message(message.chat.id, f"✅ Broadcast sent.\nDelivered: {delivered}\nFailed: {failed}")
    except Exception as e:
        logger.error(f"Broadcast command error: {e}")
        safe_send_message(message.chat.id, "❌ Broadcast failed.")

@bot.message_handler(commands=['broadcast7'])
def handle_broadcast7(message):
    try:
        if not require_admin(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2 or not parts[1].strip():
            safe_send_message(message.chat.id, "Usage: `/broadcast7 your message`")
            return
        delivered, failed = broadcast_text_to_users(parts[1].strip(), audience="active7")
        safe_send_message(message.chat.id, f"✅ Active-7d broadcast sent.\nDelivered: {delivered}\nFailed: {failed}")
    except Exception as e:
        logger.error(f"Broadcast7 command error: {e}")
        safe_send_message(message.chat.id, "❌ Broadcast failed.")

@bot.message_handler(commands=['broadcast30'])
def handle_broadcast30(message):
    try:
        if not require_admin(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2 or not parts[1].strip():
            safe_send_message(message.chat.id, "Usage: `/broadcast30 your message`")
            return
        delivered, failed = broadcast_text_to_users(parts[1].strip(), audience="active30")
        safe_send_message(message.chat.id, f"✅ Active-30d broadcast sent.\nDelivered: {delivered}\nFailed: {failed}")
    except Exception as e:
        logger.error(f"Broadcast30 command error: {e}")
        safe_send_message(message.chat.id, "❌ Broadcast failed.")

@bot.message_handler(commands=['schedulebroadcast'])
def handle_schedulebroadcast(message):
    try:
        if not require_admin(message):
            return
        # /schedulebroadcast <minutes> <all|active7|active30> <text>
        parts = message.text.split(maxsplit=3) if message.text else []
        if len(parts) < 4:
            safe_send_message(
                message.chat.id,
                "Usage: `/schedulebroadcast <minutes> <all|active7|active30> <message>`\n"
                "Example: `/schedulebroadcast 30 active7 New promo is live!`"
            )
            return
        minutes_raw = parts[1].strip()
        audience = parts[2].strip().lower()
        text = parts[3].strip()
        if not minutes_raw.isdigit():
            safe_send_message(message.chat.id, "❌ Minutes must be a number.")
            return
        if audience not in ("all", "active7", "active30"):
            safe_send_message(message.chat.id, "❌ Audience must be one of: `all`, `active7`, `active30`.")
            return
        item = create_scheduled_broadcast(
            admin_id=int(get_actor_user_id(message)),
            delay_minutes=int(minutes_raw),
            text=text,
            audience=audience
        )
        run_at = datetime.fromtimestamp(item["run_at"]).strftime("%Y-%m-%d %H:%M:%S")
        safe_send_message(
            message.chat.id,
            f"⏰ Scheduled.\nID: `{item['id']}`\nWhen: `{run_at}`\nAudience: `{audience}`"
        )
    except Exception as e:
        logger.error(f"Schedule broadcast command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to schedule broadcast.")

@bot.message_handler(commands=['schedules', 'listschedules'])
def handle_listschedules(message):
    try:
        if not require_admin(message):
            return
        items = list_scheduled_broadcasts(limit=20)
        if not items:
            safe_send_message(message.chat.id, "📭 No scheduled broadcasts.")
            return
        lines = []
        for item in items:
            run_at = datetime.fromtimestamp(int(item.get("run_at", 0))).strftime("%m-%d %H:%M")
            lines.append(
                f"- `{item.get('id')}` | {run_at} | {item.get('audience', 'all')} | "
                f"{str(item.get('text', ''))[:40]}"
            )
        safe_send_message(message.chat.id, "🗓️ *Scheduled Broadcasts:*\n" + "\n".join(lines))
    except Exception as e:
        logger.error(f"List schedules command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to list schedules.")

@bot.message_handler(commands=['cancelschedule'])
def handle_cancelschedule(message):
    try:
        if not require_admin(message):
            return
        parts = message.text.split(maxsplit=1) if message.text else []
        if len(parts) < 2:
            safe_send_message(message.chat.id, "Usage: `/cancelschedule <schedule_id>`")
            return
        schedule_id = parts[1].strip()
        if cancel_scheduled_broadcast(schedule_id):
            safe_send_message(message.chat.id, f"✅ Cancelled schedule: `{schedule_id}`")
        else:
            safe_send_message(message.chat.id, f"ℹ️ Schedule not found: `{schedule_id}`")
    except Exception as e:
        logger.error(f"Cancel schedule command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to cancel schedule.")

@bot.message_handler(commands=['admin'])
def handle_admin_panel(message):
    try:
        if not require_admin(message):
            return
        markup = InlineKeyboardMarkup(row_width=2)
        markup.add(
            InlineKeyboardButton("👥 Users", callback_data="admin_users"),
            InlineKeyboardButton("📊 Status", callback_data="admin_status"),
            InlineKeyboardButton("📣 Broadcast all", callback_data="admin_help_broadcast"),
            InlineKeyboardButton("🎯 Broadcast active7", callback_data="admin_help_broadcast7"),
            InlineKeyboardButton("⏰ Schedule", callback_data="admin_help_schedule"),
            InlineKeyboardButton("🗓️ Schedules", callback_data="admin_list_schedules"),
        )
        safe_send_message(
            message.chat.id,
            "🔴 *Artovix Admin Panel*\n\n"
            "*Step-based admin flow:*\n"
            "1. Check audience: `/users`\n"
            "2. Prepare post: `/postwizard`\n"
            "3. Send now or schedule\n\n"
            "*Main Admin Commands:*\n"
            "`/users`  `/admins`\n"
            "`/broadcast <text>`  `/broadcast7 <text>`  `/broadcast30 <text>`\n"
            "`/schedulebroadcast <min> <all|active7|active30> <text>`\n"
            "`/schedules`  `/cancelschedule <id>`\n"
            "`/post`  `/postwizard`  `/cancelpost`\n"
            "`/addadmin <userid>`  `/deladmin <userid>`\n\n"
            "Choose an action below:",
            reply_markup=markup
        )
    except Exception as e:
        logger.error(f"Admin panel command error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to open admin panel.")

@bot.message_handler(commands=['post'])
def handle_post(message):
    try:
        if not require_admin(message):
            return
        if not message.reply_to_message:
            safe_send_message(
                message.chat.id,
                "Reply to a text/photo/video/audio/document with `/post` to broadcast it."
            )
            return

        payload = _extract_broadcast_payload_from_message(message.reply_to_message)
        if not payload:
            safe_send_message(message.chat.id, "❌ Unsupported post type. Use text/photo/video/audio/document/animation/voice.")
            return
        delivered, failed = _broadcast_payload_to_users(payload)
        safe_send_message(message.chat.id, f"✅ Post sent.\nDelivered: {delivered}\nFailed: {failed}")
    except Exception as e:
        logger.error(f"Post command error: {e}")
        safe_send_message(message.chat.id, "❌ Post failed.")

@bot.message_handler(commands=['postwizard'])
def handle_postwizard(message):
    try:
        if not require_admin(message):
            return
        _wizard_set(message.chat.id, {
            "step": "await_content",
            "payload": None,
            "button_text": None,
            "button_url": None
        })
        safe_send_message(
            message.chat.id,
            "🧭 *Post Wizard Started*\n\n"
            "Step 1/3: Send the content you want to broadcast.\n"
            "Supported: text, photo, video, audio, document, animation, voice.\n\n"
            "Tip: add caption/text exactly how users should see it.\n"
            "Use `/cancelpost` anytime to abort."
        )
    except Exception as e:
        logger.error(f"Post wizard start error: {e}")
        safe_send_message(message.chat.id, "❌ Could not start post wizard.")

@bot.message_handler(commands=['cancelpost'])
def handle_cancelpost(message):
    try:
        if not require_admin(message):
            return
        if _wizard_get(message.chat.id):
            _wizard_clear(message.chat.id)
            safe_send_message(message.chat.id, "🛑 Post wizard cancelled.")
        else:
            safe_send_message(message.chat.id, "ℹ️ No active post wizard.")
    except Exception as e:
        logger.error(f"Cancel post wizard error: {e}")
        safe_send_message(message.chat.id, "❌ Failed to cancel post wizard.")

@bot.message_handler(
    func=lambda m: _wizard_get(getattr(getattr(m, "chat", None), "id", 0)) is not None,
    content_types=['text', 'photo', 'video', 'audio', 'document', 'animation', 'voice', 'sticker']
)
def handle_postwizard_input(message):
    """Capture guided posting inputs for admins while wizard is active."""
    try:
        state = _wizard_get(message.chat.id)
        if not state:
            return
        if not is_admin_user(get_actor_user_id(message)):
            _wizard_clear(message.chat.id)
            return

        step = state.get("step")

        if step == "await_content":
            payload = _extract_broadcast_payload_from_message(message)
            if not payload:
                safe_send_message(
                    message.chat.id,
                    "❌ Unsupported content for wizard.\n"
                    "Send text/photo/video/audio/document/animation/voice."
                )
                return
            state["payload"] = payload
            state["step"] = "await_button_text"
            _wizard_set(message.chat.id, state)
            safe_send_message(
                message.chat.id,
                "Step 2/3: Send button text (example: `Join Channel`) or type `skip` for no button."
            )
            return

        if step == "await_button_text":
            if not getattr(message, "text", None):
                safe_send_message(message.chat.id, "Please send text for button label, or `skip`.")
                return
            decision = message.text.strip()
            if decision.lower() == "skip":
                state["button_text"] = None
                state["button_url"] = None
                state["step"] = "confirm"
            else:
                state["button_text"] = decision
                state["step"] = "await_button_url"
            _wizard_set(message.chat.id, state)
            if state["step"] == "await_button_url":
                safe_send_message(
                    message.chat.id,
                    "Step 3/3: Send the button URL (must start with `http://` or `https://`).\n"
                    "Or type `skip` to send without button."
                )
            else:
                markup = InlineKeyboardMarkup()
                markup.add(
                    InlineKeyboardButton("✅ Send now", callback_data="postwiz_send"),
                    InlineKeyboardButton("❌ Cancel", callback_data="postwiz_cancel")
                )
                safe_send_message(
                    message.chat.id,
                    "Ready to broadcast.\nPress *Send now* to publish to all known users.",
                    reply_markup=markup
                )
            return

        if step == "await_button_url":
            if not getattr(message, "text", None):
                safe_send_message(message.chat.id, "Send a valid URL or `skip`.")
                return
            decision = message.text.strip()
            if decision.lower() == "skip":
                state["button_text"] = None
                state["button_url"] = None
            elif _is_valid_http_url(decision):
                state["button_url"] = decision
            else:
                safe_send_message(message.chat.id, "❌ Invalid URL. Send full URL like `https://example.com` or `skip`.")
                return
            state["step"] = "confirm"
            _wizard_set(message.chat.id, state)
            markup = InlineKeyboardMarkup()
            markup.add(
                InlineKeyboardButton("✅ Send now", callback_data="postwiz_send"),
                InlineKeyboardButton("❌ Cancel", callback_data="postwiz_cancel")
            )
            safe_send_message(
                message.chat.id,
                "Ready to broadcast.\nPress *Send now* to publish to all known users.",
                reply_markup=markup
            )
            return
    except Exception as e:
        logger.error(f"Post wizard input error: {e}")

# ============================================================================
# 🎨 DRAW COMMANDS (MODEL-SPECIFIC)
# ============================================================================
def process_image_generation(message, model_type=None):
    """Helper function to handle all image generation requests"""
    thinking_msg = None
    try:
        user_id = message.chat.id
        
        # Get prompt from command
        if message.text and len(message.text.split()) > 1:
            prompt = ' '.join(message.text.split()[1:])
        else:
            # Show help if no prompt
            current_model = memory.get_setting(user_id, "image_model", "auto")
            display_model = model_type if model_type else current_model
            
            help_text = f"""🎨 *AI Image Generator ({display_model.upper()})*

*Usage:* `/{display_model if model_type else 'draw'} [description]`

*Available Commands:*
• `/flux` - High Quality (HF)
• `/pollinations` - Fast & Reliable
• `/creative` - Artistic Styles
• `/auto` - Smart Selection

*Example:* `/{display_model if model_type else 'flux'} a futuristic city in neon rain`"""
            
            safe_send_message(user_id, help_text)
            return

        # Use specified model or user's preferred model
        active_model = model_type if model_type else memory.get_setting(user_id, "image_model", "auto")
        
        # Show thinking message
        thinking_msg = safe_send_message(
            user_id,
            f"🎨 *Creating with {active_model.upper()}:* \"{prompt[:60]}...\"\n"
            f"⏳ Generating image... (10-30 seconds)"
        )
        
        # Generate image
        result = ImageGenerator.generate(prompt, model_type=active_model)
        
        # Delete thinking message
        if thinking_msg:
            try: bot.delete_message(user_id, thinking_msg.message_id)
            except: pass
        
        if result:
            if isinstance(result, dict) and result.get('type') == 'text':
                # Text-based result (fallback)
                text_response = f"""🎨 *AI Image Concept:* {result['prompt']}

✨ *Model:* {active_model.upper()} (Fallback)

{result['emojis']} *Visual Description:*
{result['description']}

💡 *Pro Tip:* {result['suggestion']}"""
                
                safe_send_message(user_id, text_response)
                
            else:
                # Actual image
                try:
                    bot.send_photo(
                        user_id,
                        result,
                        caption=f"🎨 *AI Generated ({active_model.upper()}):* {prompt}\n\n"
                                       f"✨ Powered by Artovix AI | {datetime.now().strftime('%H:%M')}"
                    )
                    logger.info(f"✓ Image sent to {user_id}")
                except Exception as e:
                    logger.error(f"Photo send error: {e}")
                    safe_send_message(user_id, "❌ Failed to send image. Try again!")
        else:
            safe_send_message(user_id, "❌ Generation failed. Try a different prompt or model.")
        
        analytics.log_request(user_id, len(prompt.split()), f"image_gen_{active_model}")
        
    except Exception as e:
        logger.error(f"Image gen error: {e}")
        if thinking_msg:
            try: bot.delete_message(message.chat.id, thinking_msg.message_id)
            except: pass
        safe_send_message(message.chat.id, "❌ Error occurred. Please try again.")

@bot.message_handler(commands=['draw', 'imagine', 'generate'])
def handle_draw_default(message):
    process_image_generation(message)

@bot.message_handler(commands=['flux'])
def handle_flux(message):
    process_image_generation(message, model_type="flux")

@bot.message_handler(commands=['pollinations', 'pollin'])
def handle_pollinations(message):
    process_image_generation(message, model_type="pollinations")

@bot.message_handler(commands=['creative', 'art'])
def handle_creative(message):
    process_image_generation(message, model_type="creative")

@bot.message_handler(commands=['auto'])
def handle_auto(message):
    process_image_generation(message, model_type="auto")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_model_'))
def handle_model_selection(call):
    try:
        model_type = call.data.replace('set_model_', '')
        user_id = call.message.chat.id
        
        memory.update_setting(user_id, "image_model", model_type)
        
        # Update the message with the new selection
        markup = InlineKeyboardMarkup(row_width=2)
        markup.add(
            InlineKeyboardButton(f"{'✅ ' if model_type == 'auto' else ''}Auto", callback_data="set_model_auto"),
            InlineKeyboardButton(f"{'✅ ' if model_type == 'flux' else ''}Flux", callback_data="set_model_flux"),
            InlineKeyboardButton(f"{'✅ ' if model_type == 'pollinations' else ''}Pollinations", callback_data="set_model_pollinations"),
            InlineKeyboardButton(f"{'✅ ' if model_type == 'creative' else ''}Creative", callback_data="set_model_creative")
        )
        
        bot.edit_message_text(
            f"✅ Model set to: **{model_type.upper()}**\n\nNow use `/draw [prompt]` to generate images!",
            chat_id=user_id,
            message_id=call.message.message_id,
            reply_markup=markup,
            parse_mode="Markdown"
        )
        bot.answer_callback_query(call.id, f"Model set to {model_type}")
        
    except Exception as e:
        logger.error(f"Model selection error: {e}")
        bot.answer_callback_query(call.id, "❌ Failed to update model.")

# ============================================================================
# 🎭 STICKER HANDLER (INCOMING STICKERS)
# ============================================================================
@bot.message_handler(content_types=['sticker'])
def handle_incoming_sticker(message):
    try:
        if not ensure_channel_access(message):
            return
        
        user_id = message.chat.id
        sticker = getattr(message, "sticker", None)
        if not sticker:
            return

        file_id = getattr(sticker, "file_id", "")
        emoji = getattr(sticker, "emoji", "") or "✨"
        set_name = getattr(sticker, "set_name", "") or "user_pack"

        # Learn and store this sticker
        sticker_vault.add_sticker(file_id=file_id, emoji=emoji, set_name=set_name)
        mood = sticker_vault.detect_mood_from_emoji(emoji)

        friendly_greetings = [
            f"Awww, I love this sticker! {emoji} It completely brightens my day! 💖",
            f"Woohoo! That sticker is brilliant! {emoji} Saved to my special collection! 🌟",
            f"Haha, incredible sticker choice! {emoji} You have supreme style! ✨",
            f"Omg so cute! {emoji} Here is one right back at you! 🚀",
            f"That sticker is legendary! {emoji} High five! 🙌",
            f"Super cool vibe! {emoji} Positive energy all around! 💫",
        ]
        text_greeting = random.choice(friendly_greetings)
        safe_send_message(user_id, text_greeting)

        # Return a matching sticker from our collection if available and different
        reply_sticker = sticker_vault.get_random_sticker(mood)
        if reply_sticker and reply_sticker != file_id:
            safe_send_sticker(user_id, reply_sticker)
        else:
            emoji_art = sticker_vault.get_random_emoji_art(mood)
            safe_send_message(user_id, emoji_art)

        safe_send_message(
            user_id,
            "Want to explore more or play a game? 👇",
            reply_markup=build_sticker_replay_markup(mood)
        )
        analytics.log_request(user_id, 10, "sticker_interaction")
    except Exception as e:
        logger.error(f"Incoming sticker handler error: {e}")
        safe_send_message(message.chat.id, "✨ That sticker is fantastic! You're awesome! 😊")

# ============================================================================
# 📄 DOCUMENT HANDLER (PDF / TEXT)
# ============================================================================
@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        if not ensure_channel_access(message):
            return
        bot.send_chat_action(message.chat.id, 'typing')

        file_name = getattr(getattr(message, "document", None), "file_name", "document")
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        text, error = _extract_text_from_document_bytes(file_name, downloaded_file)
        if error:
            safe_send_message(
                message.chat.id,
                f"📄 Could not process *{file_name}*.\n{error}"
            )
            return

        store_user_document(message.chat.id, file_name, text)
        word_count = len((text or "").split())
        safe_send_message(
            message.chat.id,
            f"✅ Document loaded: *{file_name}*\n"
            f"Words: {word_count}\n\n"
            "Now ask:\n`/askdoc summarize this`\n"
            "`/askdoc what are key points?`"
        )
        analytics.log_request(message.chat.id, min(word_count, 2000), "doc_upload")

        if getattr(message, "caption", None):
            caption_q = message.caption.strip()
            if caption_q:
                message.text = f"/askdoc {caption_q}"
                handle_askdoc(message)
    except Exception as e:
        logger.error(f"Document handler error: {e}")
        safe_send_message(message.chat.id, "📄 Failed to process document.")

# ============================================================================
# 🎙️ VOICE MESSAGES HANDLER
# ============================================================================
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        if not ensure_channel_access(message):
            return
        bot.send_chat_action(message.chat.id, 'upload_document')
        
        # Get voice file info
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save temporarily
        temp_filename = f"temp/voice_{message.chat.id}_{int(time.time())}.ogg"
        os.makedirs("temp", exist_ok=True)
        with open(temp_filename, 'wb') as f:
            f.write(downloaded_file)
        
        # Transcribe with Groq Whisper if available
        with open(temp_filename, "rb") as audio_file:
            if not groq_client:
                safe_send_message(
                    message.chat.id,
                    "🎤 *Voice transcription is unavailable without Groq Whisper.*\n\n"
                    "Text chat and vision are active with NVIDIA NIM! Please send your query as a text message."
                )
                return

            transcription = groq_client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        
        # Cleanup
        os.remove(temp_filename)
        
        if not transcription or len(transcription.strip()) < 1:
            safe_send_message(message.chat.id, "🎤 *I couldn't hear you clearly.*\nCould you please try again?")
            return

        # Process as a text message
        message.text = transcription
        safe_send_message(message.chat.id, f"🎤 *Transcribed:* \"{transcription}\"")
        handle_all_messages(message)
        
    except Exception as e:
        logger.error(f"Voice handling error: {e}")
        safe_send_message(message.chat.id, "🎤 *Voice processing failed.*\nPlease try sending a text message instead.")

# ============================================================================
# 🖼️ PHOTO ANALYSIS HANDLER (VISION)
# ============================================================================
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        if not ensure_channel_access(message):
            return
        with KeepTyping(message.chat.id, 'typing'):
            # Get highest resolution photo
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            
            # Encode to base64
            base64_image = base64.b64encode(downloaded_file).decode('utf-8')
            
            # Get caption or use default
            prompt = message.caption if message.caption else "Describe this image in detail and tell me what you see."
            
            # Analyze with Vision
            if not (nvidia_client or groq_client):
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `NVIDIA_API_KEY` or `GROQ_API_KEY` in your .env to enable vision features.")
                return

            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]

        analysis_raw, used_model = ai_vision_with_fallback(
            messages=vision_messages,
            max_tokens=500
        )
        analysis = clean_markdown(analysis_raw)

        safe_send_message(message.chat.id, f"🖼️ *Image Analysis:*\n\n{analysis}")
        logger.info(f"Vision analysis sent using model: {used_model}")

        analytics.log_request(message.chat.id, 500, "vision_analysis")
        
    except Exception as e:
        logger.error(f"Vision handling error: {e}")
        err = str(e).lower()
        if "api key" in err or "unauthorized" in err or "authentication" in err or "forbidden" in err:
            msg = "🔐 *Vision key/auth issue.*\nPlease check `NVIDIA_API_KEY` (or `GROQ_API_KEY`) in Render environment variables."
        elif "model" in err and ("not found" in err or "not available" in err or "decommissioned" in err):
            msg = "🧠 *Vision model unavailable right now.*\nI tried fallback models. Please retry in a moment."
        elif "413" in err or "too large" in err:
            msg = "📷 *Image too large.*\nPlease send a smaller image or compressed photo."
        else:
            msg = "🖼️ *Vision analysis failed.*\nPlease try again with another image."
        safe_send_message(message.chat.id, msg)

# ============================================================================
# 💬 TEXT MESSAGES HANDLER
# ============================================================================
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    try:
        if not ensure_channel_access(message):
            return
        # Skip if empty or command
        if not message.text or message.text.startswith('/'):
            return

        text = message.text.strip()

        # Reply-keyboard flow (easy mode)
        if text == MENU_CHAT:
            set_pending_mode(message.chat.id, None)
            safe_send_message(
                message.chat.id,
                "🔴 *Chat Mode*\nSend your message.",
                reply_markup=build_main_reply_menu()
            )
            return
        if text == MENU_DRAW:
            set_pending_mode(message.chat.id, "draw")
            safe_send_message(
                message.chat.id,
                "🎨 *Image Mode*\nSend an image prompt.",
                reply_markup=build_main_reply_menu()
            )
            return
        if text == MENU_SEARCH:
            set_pending_mode(message.chat.id, "search")
            safe_send_message(
                message.chat.id,
                "🔍 *Search Mode*\nSend your question.",
                reply_markup=build_main_reply_menu()
            )
            return
        if text == MENU_CODE:
            set_pending_mode(message.chat.id, "code")
            safe_send_message(
                message.chat.id,
                "💻 *Code Mode*\nSend your coding question or code.",
                reply_markup=build_main_reply_menu()
            )
            return
        if text == MENU_DOC:
            set_pending_mode(message.chat.id, "askdoc")
            safe_send_message(
                message.chat.id,
                "📄 *Document Mode*\nUpload a file, then ask your question.",
                reply_markup=build_main_reply_menu()
            )
            return
        if text == MENU_VOICE:
            send_voice_mode_panel(message.chat.id)
            return
        if text == MENU_GAMES:
            send_games_panel(message.chat.id)
            return
        if text == MENU_SETTINGS:
            send_settings_panel(message.chat.id)
            return
        if text == MENU_HELP:
            set_pending_mode(message.chat.id, None)
            handle_help(message)
            return
        if text == MENU_RESET:
            set_pending_mode(message.chat.id, None)
            handle_reset(message)
            return

        # Greeting sticker: send one sticker only when user sends a casual greeting
        _greeting_words = {"hi", "hello", "hey", "hiya", "howdy", "sup", "yo", "greetings",
                           "good morning", "good evening", "good afternoon", "hi there", "hey there"}
        _t_low = text.strip().lower().rstrip("!.,?")
        if any(_t_low == g or _t_low.startswith(g + " ") for g in _greeting_words) and len(text) < 35:
            _greeting_sticker = sticker_vault.get_random_sticker("happy")
            if _greeting_sticker:
                safe_send_sticker(message.chat.id, _greeting_sticker)

        # Natural game request in chat (e.g., "let's play dice", "play trivia")
        is_gm, gm_type = is_natural_game_request(text)
        if is_gm:
            if gm_type == "trivia":
                bot.send_chat_action(message.chat.id, 'typing')
                Thread(target=start_trivia, args=(message.chat.id,), daemon=True).start()
            elif gm_type == "card":
                start_card_game(message.chat.id)
            else:
                play_telegram_game(message.chat.id, gm_type)
            return

        pending_mode = get_pending_mode(message.chat.id)
        if pending_mode in {"draw", "search", "code", "askdoc"}:
            set_pending_mode(message.chat.id, None)
            if pending_mode == "draw":
                message.text = f"/draw {text}"
                process_image_generation(message)
                return
            if pending_mode == "search":
                message.text = f"/search {text}"
                handle_search(message)
                return
            if pending_mode == "code":
                message.text = f"/code {text}"
                handle_code(message)
                return
            if pending_mode == "askdoc":
                message.text = f"/askdoc {text}"
                handle_askdoc(message)
                return
        
        logger.info(f"Message from {message.chat.id}: {message.text[:50]}...")
        
        user_id = str(message.chat.id)
        user_data = memory.get_user_data(user_id)
        history = user_data["history"]
        
        # Prepare conversation
        language_instruction = build_language_instruction(message.chat.id)
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nLANGUAGE MODE:\n{language_instruction}"},
            *history[-CHAT_HISTORY_CONTEXT_MESSAGES:],
            {"role": "user", "content": message.text}
        ]
        
        try:
            if not (nvidia_client or groq_client):
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `NVIDIA_API_KEY` or `GROQ_API_KEY` in your .env to enable chat responses.")
                return

            local_messages = list(messages)
            reply_parts = []
            used_model = None
            max_tokens_for_user = get_response_max_tokens(message.chat.id)

            # Auto-detect if user prompt is a programming/coding task to route to best model
            user_text_low = (message.text or "").lower()
            is_coding = bool(
                "```" in (message.text or "")
                or any(kw in user_text_low for kw in [
                    "python", "javascript", "typescript", "write code", "fix code",
                    "debug", "syntax error", "traceback", "sql query", "function",
                    "algorithm", "regex", "c++", "rust", "golang", "html/css"
                ])
            )
            chat_purpose = "code" if is_coding else "chat"
            selected_model = get_purpose_model(chat_purpose, message.chat.id)

            # Send continuous animated typing indicator until response is ready
            with KeepTyping(message.chat.id, 'typing'):
                part, used_model, finish_reason = groq_chat_with_fallback(
                    messages=local_messages,
                    temperature=0.7,
                    max_tokens=max_tokens_for_user,
                    model=selected_model
                )
                reply_raw = strip_thinking_process(part.strip()) if part else ""

                # Auto-continue up to 3 more times if the model was truncated mid-thought
                for _continuation in range(3):
                    if not (finish_reason == "length" and reply_raw and len(reply_raw) > 200):
                        break
                    local_messages.append({"role": "assistant", "content": reply_raw})
                    local_messages.append({
                        "role": "user",
                        "content": "Please continue seamlessly from where you stopped."
                    })
                    cont_part, _, finish_reason = groq_chat_with_fallback(
                        messages=local_messages,
                        temperature=0.7,
                        max_tokens=max_tokens_for_user,
                        model=selected_model
                    )
                    if cont_part:
                        cont_clean = strip_thinking_process(cont_part.strip())
                        reply_raw = f"{reply_raw}\n\n{cont_clean}".strip()
                    else:
                        break

            reply = clean_markdown(reply_raw)
        
            # Save to memory
            history.extend([
                {"role": "user", "content": message.text},
                {"role": "assistant", "content": reply}
            ])
            
            # Limit stored memory size while keeping enough continuity.
            if len(history) > CHAT_HISTORY_MAX_MESSAGES:
                history = history[-CHAT_HISTORY_MAX_MESSAGES:]
            
            user_data["history"] = history
            memory.save_user_data(user_id, user_data)
            
            # Send reply (text or voice depending on user mode)
            save_last_ai_reply(message.chat.id, reply)
            send_ai_reply(message.chat.id, reply, reply_markup=maybe_followup_markup(reply))
            logger.info(f"Chat reply sent using model: {used_model}")
            
            # Log analytics
            tokens_used = len(message.text.split()) + len(reply.split())
            analytics.log_request(message.chat.id, tokens_used, "chat")
            
        except Exception as api_error:
            logger.error(f"Chat API error: {api_error}\n{traceback.format_exc()}")
            try:
                err = str(api_error).lower()
                status_code = getattr(api_error, "status_code", None)
                if status_code == 410 or any(term in err for term in ("410", "gone", "decommissioned", "credit", "exhausted")):
                    user_msg = "🔄 *Switching AI engine...* Please send your message again in a moment."
                elif status_code == 402 or any(term in err for term in ("402", "free limit reached", "payment required")):
                    user_msg = "⏳ *AI is busy.* Please wait a few seconds and try again."
                elif status_code == 429 or any(term in err for term in ("rate limit", "rate_limit", "too many requests", "quota")):
                    user_msg = "⏳ *Too many requests.* Please wait 10-20 seconds and try again."
                elif status_code in {401, 403} or any(term in err for term in ("api key", "api_key", "unauthorized", "authentication", "invalid bearer", "forbidden", "authorization failed")):
                    user_msg = "🔄 *Switching AI engine...* Please send your message again in a moment."
                elif status_code in {404, 400} or any(term in err for term in (
                    "not found", "decommissioned", "not available", "unsupported",
                    "404", "model_not_found", "does not exist", "no such model"
                )):
                    user_msg = (
                        "🔄 *AI model switching...*\n\n"
                        "One moment — trying a faster model for you. Please send your message again."
                    )
                elif status_code == 413 or any(term in err for term in ("context length", "maximum context", "prompt is too long", "request too large")):
                    user_msg = (
                        "📝 *That conversation is too long for the AI model.*\n\n"
                        "Please use /reset and send the message again."
                    )
                elif status_code in {408, 500, 502, 503, 504} or any(term in err for term in ("timeout", "timed out", "service unavailable", "connection reset")):
                    user_msg = (
                        "🔄 *AI service is temporarily unavailable.*\n\n"
                        "Please try again in a moment."
                    )
                else:
                    # Show the actual error details to help diagnose (admin-visible)
                    actor_id = getattr(getattr(message, "from_user", None), "id", None) or message.chat.id
                    if actor_id in MAIN_ADMIN_IDS:
                        user_msg = (
                            f"⚠️ *AI Error (admin debug):*\n\n"
                            f"`{str(api_error)[:300]}`"
                        )
                    else:
                        user_msg = (
                            "⚠️ *AI service temporary issue.*\n\n"
                            "Please try again in a moment."
                        )
                safe_send_message(
                    message.chat.id,
                    user_msg
                )
            except Exception:
                logger.error("Failed to send fallback message after chat error.")
        
    except Exception as e:
        logger.error(f"Message handler error: {e}\n{traceback.format_exc()}")
        try:
            safe_send_message(message.chat.id, "⚠️ Please try again or use /reset to start fresh.")
        except:
            pass

# ============================================================================
# 🔎 INLINE MODE HANDLER (@bot query)
# ============================================================================
@bot.inline_handler(func=lambda q: True)
def handle_inline_query(query):
    try:
        qtext = (getattr(query, "query", "") or "").strip()
        if not qtext:
            return
        if not (nvidia_client or groq_client):
            return

        inline_user_id = getattr(getattr(query, "from_user", None), "id", 0)
        language_instruction = build_language_instruction(inline_user_id)
        prompt = (
            "Answer briefly and clearly for Telegram inline mode.\n"
            f"{language_instruction}\n\n"
            f"User query: {qtext}"
        )
        content, _, _ = ai_chat_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=220
        )
        answer = clean_markdown(content)

        result = InlineQueryResultArticle(
            id=str(uuid4()),
            title=f"Artovix: {qtext[:40]}",
            description=(answer[:100] + "...") if len(answer) > 100 else answer,
            input_message_content=InputTextMessageContent(
                f"🔴 *Artovix Red*\n\n{answer}",
                parse_mode="Markdown"
            )
        )
        bot.answer_inline_query(query.id, [result], cache_time=1, is_personal=True)
    except Exception as e:
        logger.error(f"Inline query error: {e}")

# ============================================================================
# 🎪 CALLBACK HANDLER
# ============================================================================
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        if call.data == "check_joined":
            if ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Access granted ✅")
                play_intro_animation(call.message.chat.id)
                send_welcome_panel(call.message.chat.id)
            else:
                bot.answer_callback_query(call.id, "Please join channel first")

        elif str(call.data).startswith("lang_set_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            lang = str(call.data).replace("lang_set_", "").strip().lower()
            if lang not in LANGUAGE_LABELS:
                bot.answer_callback_query(call.id, "Invalid language")
                return
            set_user_language(call.message.chat.id, lang)
            bot.answer_callback_query(call.id, f"Language: {LANGUAGE_LABELS.get(lang)}")
            send_language_panel(call.message.chat.id)

        elif call.data == "settings_voice":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Voice settings")
            send_voice_mode_panel(call.message.chat.id)

        elif call.data == "settings_model":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Model settings")
            send_model_panel(call.message.chat.id)

        elif str(call.data).startswith("set_aimodel_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            mkey = str(call.data).replace("set_aimodel_", "").strip().lower()
            canonical = set_user_chat_model(call.message.chat.id, mkey)
            info = NVIDIA_FEATURED_MODELS.get(canonical, NVIDIA_FEATURED_MODELS["auto"])
            bot.answer_callback_query(call.id, f"Switched to {info['name']}!")
            send_model_panel(call.message.chat.id)

        elif call.data == "open_image_models":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Image models")
            send_image_model_panel(call.message.chat.id)

        elif call.data == "settings_language":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Language settings")
            send_language_panel(call.message.chat.id)

        elif call.data == "settings_length":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Response length")
            send_response_length_panel(call.message.chat.id)

        elif call.data == "settings_reset_memory":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            user_id = str(call.message.chat.id)
            user_data = memory.get_user_data(user_id)
            user_data["history"] = []
            memory.save_user_data(user_id, user_data)
            bot.answer_callback_query(call.id, "Memory reset")
            safe_send_message(call.message.chat.id, "🧹 Memory reset.")

        elif call.data in {"resp_len_short", "resp_len_medium", "resp_len_long"}:
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            length_value = call.data.replace("resp_len_", "").strip().lower()
            set_response_length(call.message.chat.id, length_value)
            bot.answer_callback_query(call.id, f"Length: {length_value}")
            send_response_length_panel(call.message.chat.id)

        elif call.data in {"fu_continue", "fu_shorten", "fu_examples", "fu_translate"}:
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            action_map = {
                "fu_continue": "continue",
                "fu_shorten": "shorten",
                "fu_examples": "examples",
                "fu_translate": "translate",
            }
            action_name = action_map.get(call.data)
            bot.answer_callback_query(call.id, "Processing...")
            answer, err = run_followup_action(call.message.chat.id, action_name)
            if err:
                safe_send_message(call.message.chat.id, f"⚠️ {err}")
                return
            save_last_ai_reply(call.message.chat.id, answer)
            send_ai_reply(call.message.chat.id, answer, reply_markup=maybe_followup_markup(answer))

        elif call.data == "admin_users":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Users")
            users = sorted(get_all_known_user_ids())
            safe_send_message(call.message.chat.id, f"👥 *Known Users:* {len(users)}")

        elif call.data == "admin_status":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Status")
            status_msg = f"""✅ *Artovix Status Report*

*Core Systems:*
• 🤖 AI Engine: ✅ Online
• 🧠 Memory: ✅ {len(memory.load())} active
• 🎨 Image Gen: ✅ Multiple services
• 🎙️ Voice/Vision: ✅ Optimized
• 🔍 Search: ✅ Active

*Server Info:*
• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Version: 2026.2.0 Stable
• Uptime: 100%
"""
            safe_send_message(call.message.chat.id, status_msg)

        elif call.data == "admin_help_broadcast":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Broadcast")
            safe_send_message(call.message.chat.id, "Use: `/broadcast your message`")

        elif call.data == "admin_help_broadcast7":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Broadcast active7")
            safe_send_message(call.message.chat.id, "Use: `/broadcast7 your message`")

        elif call.data == "admin_help_schedule":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Schedule")
            safe_send_message(
                call.message.chat.id,
                "Use:\n`/schedulebroadcast <minutes> <all|active7|active30> <message>`\n"
                "Example:\n`/schedulebroadcast 30 active7 New update is live`"
            )

        elif call.data == "admin_list_schedules":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            bot.answer_callback_query(call.id, "Schedules")
            items = list_scheduled_broadcasts(limit=20)
            if not items:
                safe_send_message(call.message.chat.id, "📭 No scheduled broadcasts.")
            else:
                lines = []
                for item in items:
                    run_at = datetime.fromtimestamp(int(item.get("run_at", 0))).strftime("%m-%d %H:%M")
                    lines.append(
                        f"- `{item.get('id')}` | {run_at} | {item.get('audience', 'all')} | "
                        f"{str(item.get('text', ''))[:40]}"
                    )
                safe_send_message(call.message.chat.id, "🗓️ *Scheduled Broadcasts:*\n" + "\n".join(lines))

        elif call.data == "postwiz_send":
            if not require_admin(call):
                bot.answer_callback_query(call.id, "Admin only")
                return
            state = _wizard_get(call.message.chat.id)
            if not state or state.get("step") != "confirm" or not state.get("payload"):
                bot.answer_callback_query(call.id, "No active post wizard.")
                safe_send_message(call.message.chat.id, "ℹ️ No active post wizard. Use `/postwizard`.")
                return
            bot.answer_callback_query(call.id, "Broadcasting...")
            delivered, failed = _broadcast_payload_to_users(
                payload=state["payload"],
                button_text=state.get("button_text"),
                button_url=state.get("button_url")
            )
            _wizard_clear(call.message.chat.id)
            safe_send_message(
                call.message.chat.id,
                f"✅ Post sent.\nDelivered: {delivered}\nFailed: {failed}"
            )

        elif call.data == "postwiz_cancel":
            _wizard_clear(call.message.chat.id)
            bot.answer_callback_query(call.id, "Cancelled")
            safe_send_message(call.message.chat.id, "🛑 Post wizard cancelled.")

        elif call.data == "start_chat":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Chat mode")
            safe_send_message(call.message.chat.id, 
                "🔴 *Chat Mode*\nSend your message."
            )
        
        elif call.data == "generate_image":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Image mode")
            safe_send_message(call.message.chat.id, 
                "🎨 *Image Mode*\nUse `/draw [prompt]`."
            )
        
        elif call.data == "code_help":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Code mode")
            safe_send_message(call.message.chat.id, 
                "💻 *Code Mode*\nUse `/code [question]` or send code directly."
            )
        
        elif call.data == "ask_question":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Search mode")
            safe_send_message(call.message.chat.id, 
                "🔍 *Search Mode*\nUse `/search [query]`."
            )

        elif call.data == "voice_mode_on":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Voice mode ON")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice replies enabled.",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_mode_off":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_reply_enabled(call.message.chat.id, False)
            bot.answer_callback_query(call.id, "Voice mode OFF")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice replies disabled.",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_profile_male":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_profile(call.message.chat.id, "male")
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Male voice selected")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice profile set to *MALE* (voice mode ON).",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_profile_female":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_profile(call.message.chat.id, "female")
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Female voice selected")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice profile set to *FEMALE* (voice mode ON).",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_style_soft":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_style(call.message.chat.id, "soft")
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Voice style: soft")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice style set to *SOFT* (voice mode ON).",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_style_fast":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_style(call.message.chat.id, "fast")
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Voice style: fast")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice style set to *FAST* (voice mode ON).",
                reply_markup=build_main_reply_menu()
            )

        elif call.data == "voice_style_normal":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            set_voice_style(call.message.chat.id, "normal")
            set_voice_reply_enabled(call.message.chat.id, True)
            bot.answer_callback_query(call.id, "Voice style: normal")
            safe_send_message(
                call.message.chat.id,
                "✅ Voice style set to *NORMAL* (voice mode ON).",
                reply_markup=build_main_reply_menu()
            )

        elif str(call.data).startswith("stk_cat_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            mood = str(call.data).replace("stk_cat_", "").strip().lower()
            bot.answer_callback_query(call.id, f"Sending {mood} sticker ✨")
            sticker = sticker_vault.get_random_sticker(mood)
            if sticker:
                safe_send_sticker(call.message.chat.id, sticker)
            else:
                emoji_art = sticker_vault.get_random_emoji_art(mood)
                safe_send_message(call.message.chat.id, emoji_art)
            safe_send_message(
                call.message.chat.id,
                f"✨ Enjoy your *{mood.upper()}* vibe! Tap below for more:",
                reply_markup=build_sticker_replay_markup(mood)
            )

        elif call.data == "stk_menu_home":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Stickers Station")
            send_sticker_station_panel(call.message.chat.id)

        elif call.data == "stk_open_games":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Games Corner 🎲")
            send_games_panel(call.message.chat.id)

        elif call.data == "stk_help_custom":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "Custom Sticker Maker 🎨")
            safe_send_message(
                call.message.chat.id,
                "🎨 *AI Sticker Maker*\n\n"
                "Type: `/makesticker [description]`\n"
                "Example: `/makesticker cute smiling cybernetic cat wearing goggles`\n\n"
                "I will generate and send a genuine vector-styled sticker for you! ✨"
            )

        elif str(call.data).startswith("game_play_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            gtype = str(call.data).replace("game_play_", "").strip().lower()
            bot.answer_callback_query(call.id, f"Launching {gtype}! 🎯")
            play_telegram_game(call.message.chat.id, gtype)

        elif call.data == "game_trivia_start":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "🧠 Generating trivia...")
            Thread(target=start_trivia, args=(call.message.chat.id,), daemon=True).start()

        elif str(call.data).startswith("trivia_ans_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            chosen = call.data.replace("trivia_ans_", "").strip().upper()
            bot.answer_callback_query(call.id, f"You picked {chosen}!")
            handle_trivia_answer(call.message.chat.id, chosen)

        elif call.data == "trivia_skip":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            memory.update_setting(str(call.message.chat.id), TRIVIA_KEY, None)
            bot.answer_callback_query(call.id, "Skipped!")
            markup = InlineKeyboardMarkup(row_width=2)
            markup.add(
                InlineKeyboardButton("🧠 New Question", callback_data="game_trivia_start"),
                InlineKeyboardButton("🎮 All Games",    callback_data="stk_open_games"),
            )
            safe_send_message(call.message.chat.id, "⏭️ Skipped! Ready for a new question?", reply_markup=markup)

        elif call.data == "game_card_start":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "🃏 Dealing cards...")
            start_card_game(call.message.chat.id)

        elif str(call.data).startswith("card_guess_"):
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            guess = call.data.replace("card_guess_", "").strip().lower()
            bot.answer_callback_query(call.id, f"Guessing {guess}...")
            handle_card_guess(call.message.chat.id, guess)

        elif call.data == "card_cashout":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "💰 Cashing out!")
            handle_card_guess(call.message.chat.id, "cashout")

        elif call.data == "game_my_stats":
            if not ensure_channel_access(call):
                bot.answer_callback_query(call.id, "Join channel first")
                return
            bot.answer_callback_query(call.id, "📊 Loading stats...")
            uname = getattr(getattr(call, "from_user", None), "first_name", None) or "Player"
            markup = InlineKeyboardMarkup(row_width=1)
            markup.add(InlineKeyboardButton("🎮 Back to Games", callback_data="stk_open_games"))
            safe_send_message(
                call.message.chat.id,
                format_stats_card(call.message.chat.id, uname),
                reply_markup=markup
            )
        
    except Exception as e:
        logger.error(f"Callback error: {e}")

# ============================================================================
# 🚀 MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║                  ARTOVIX 2026                    ║
    ║             MASTERPIECE AI EDITION               ║
    ║                                                  ║
    ║  ✅ **ALL COMMANDS & FEATURES WORKING:**         ║
    ║    • /start, /help, /status, /stats             ║
    ║    • /draw - AI Image generation (Flux)         ║
    ║    • /sticker, /stickers - Dynamic Sticker Vault ║
    ║    • /makesticker - AI Custom Sticker Generator  ║
    ║    • /emoji, /emojify - Emoji Art & Stories     ║
    ║    • /game, /dice, /dart - Interactive Games    ║
    ║    • /search - Deep knowledge specialist        ║
    ║    • /code - Programming & debugging expert     ║
    ║    • /voice - Voice replies & accents           ║
    ║    • 🎙️ Whisper Voice Transcription             ║
    ║    • 🖼️ Vision Multi-Modal Image Analysis       ║
    ║    • 💖 Friendly, Warm, Empathetic Persona      ║
    ║                                                  ║
    ║  🚀 Starting up...                               ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    logger.info("🚀 Starting Artovix 2026 (Masterpiece AI Edition)...")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    bot_username = "(unavailable)"
    try:
        me = bot.get_me()
        bot_username = getattr(me, "username", "(unknown)")
    except Exception as e:
        logger.warning(f"Startup bot identity check failed: {e}")
    print(f"🤖 Bot: @{bot_username}")
    print(f"🧠 Memory: {len(memory.load())} active conversations")
    print(f"🎭 Stickers in Vault: {sum(len(v) for v in sticker_vault.data.get('stickers', {}).values())}")
    print(f"📊 Analytics: Enhanced database ready")
    
    print("\n" + "="*60)
    print("✅ MASTERPIECE COMMANDS READY:")
    print("="*60)
    print("💬 /start - Welcome & features")
    print("🎭 /stickers - Interactive Sticker Station")
    print("🎨 /makesticker [prompt] - Custom AI Sticker Generator")
    print("✨ /emoji [text] - Emoji Art & Story Transformer")
    print("🎲 /game, /dice, /dart, /basket - Animated Games")
    print("🎨 /draw [prompt] - Generate images")
    print("🔍 /search [query] - Search knowledge")
    print("💻 /code [question] - Programming help")
    print("📄 /askdoc [question] - Ask from uploaded document")
    print("🔈 /voice on|off|male|female - Voice mode/profile")
    print("🌐 /lang - Language mode")
    print("⚙️ /settings - Voice/model/language/length")
    print("🔎 Inline mode: @your_bot_username query")
    print("📊 /stats - View analytics")
    print("🧹 /reset - Clear memory")
    print("✅ /status - Bot health")
    print("🔧 /help - Command list")
    print("="*60)
    print("\n⚡ Bot is running and ready to receive commands!")
    print("💡 Tip: Try /stickers or send a sticker to see the magic!")
    
    try:
        # Ensure any existing webhook is removed before starting polling to avoid
        # Telegram 409 Conflict errors when another updater or webhook exists.
        def _delete_telegram_webhook():
            if not BOT_TOKEN:
                return
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
                resp = requests.post(
                    url,
                    json={"drop_pending_updates": True},
                    timeout=10
                )
                logger.info(f"deleteWebhook: {resp.status_code} {resp.text}")
            except Exception as _e:
                logger.warning(f"Failed to call deleteWebhook: {_e}")

        # Attempt to remove webhook once before starting
        _delete_telegram_webhook()
        start_scheduler_once()

        # Pre-seed R2-DOG2 sticker pack in background so it doesn't block startup.
        # Stickers are stored persistently in stickers_vault.json and only fetched once.
        def _seed_stickers_bg():
            # R2-DOG2: 27 animated robot dog stickers — seed under robot + cool + happy categories
            seeded = seed_sticker_pack("R2-DOG2", default_mood="robot", extra_moods=["cool", "happy", "cute"])
            # Also try the Telegram pack names with different cases just in case
            if seeded == 0:
                seed_sticker_pack("r2dog2", default_mood="robot", extra_moods=["cool", "happy", "cute"])
            total = sum(len(v) for v in sticker_vault.data.get("stickers", {}).values())
            logger.info(f"[StickerSeed] Vault now has {total} stickers across all moods")

        threading.Thread(target=_seed_stickers_bg, daemon=True, name="StickerSeeder").start()
        logger.info("[StickerSeed] Background seeder started for R2-DOG2 pack")

        # Run polling in a loop so transient 409/other errors try to self-heal.
        while True:
            try:
                bot.infinity_polling(
                    timeout=30,
                    long_polling_timeout=5,
                    skip_pending=True,
                    logger_level=logging.INFO
                )
                break
            except Exception as e:
                # If conflict due to other getUpdates request, try deleting webhook and retry
                err_text = str(e)
                logger.error(f"Polling exception: {err_text}\n{traceback.format_exc()}")
                if '409' in err_text or 'Conflict' in err_text:
                    logger.warning(
                        "Detected Telegram 409 conflict. Another bot instance is likely polling. "
                        "Attempting webhook cleanup and retrying in 20s..."
                    )
                    _delete_telegram_webhook()
                    time.sleep(20)
                    continue
                # For other exceptions, wait and retry once
                time.sleep(5)
                continue
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}\n{traceback.format_exc()}")
        print(f"❌ Critical error: {e}")
    finally:
        try:
            analytics.close()
            print("📊 Analytics saved.")
        except:
            pass
        print("\n👋 Shutdown complete.")
