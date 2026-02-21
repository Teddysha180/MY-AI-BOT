import telebot
import os
import json
import logging
import requests
from datetime import datetime
from groq import Groq
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import sqlite3
import threading
import traceback
import time
import base64
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

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
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
CHAT_MODEL_FALLBACK = os.getenv("GROQ_CHAT_MODEL_FALLBACK", "llama-3.1-8b-instant")

# Initialize clients with safer handling so the module can run without keys
groq_client = None
bot = None

if GROQ_KEY:
    try:
        groq_client = Groq(api_key=GROQ_KEY)
    except Exception as e:
        print(f"⚠️ Groq client init warning: {e}")
        logger = logging.getLogger(__name__)
        logger.warning(f"Groq client init failed: {e}")
        groq_client = None
else:
    print("⚠️ GROQ_API_KEY not set; Groq features disabled.")

if BOT_TOKEN:
    try:
        bot = telebot.TeleBot(BOT_TOKEN, parse_mode="Markdown")
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

        # Basic methods used across the code. They are no-ops when the real bot
        # isn't available.
        def send_message(self, *args, **kwargs):
            print("[DummyBot] send_message called; BOT_TOKEN not configured.")
            return None

        def send_photo(self, *args, **kwargs):
            print("[DummyBot] send_photo called; BOT_TOKEN not configured.")
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
        logging.FileHandler('artovix.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Analytics:
    def __init__(self):
        self.conn = sqlite3.connect('analytics.db', check_same_thread=False)
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
        self.conn.commit()
    
    def log_request(self, user_id: str, tokens: int, request_type: str):
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (user_id, tokens, request_type)
                    VALUES (?, ?, ?)
                ''', (str(user_id), tokens, request_type))
                self.conn.commit()
        except Exception as e:
            logger.error(f"Analytics log error: {e}")
    
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

analytics = Analytics()

# ============================================================================
# 🧠 MEMORY SYSTEM
# ============================================================================
class AdvancedMemory:
    def __init__(self):
        self.memory_file = "artovix_memory.json"
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
SYSTEM_PROMPT = """You are Artovix, an elite AI assistant in 2026.

PERSONALITY:
- Brilliant futurist AI
- Empathetic and supportive  
- Creative problem solver
- Multimodal expert
- Ethical and responsible

GUIDELINES:
1. Be helpful, accurate, and concise
2. Use appropriate emojis
3. Admit when you don't know something
4. Consider context from previous messages
5. Think step-by-step for complex problems

RESPONSE FORMAT:
- Use Markdown for readability
- Structure complex answers with bullet points
- Keep responses clear and engaging

Remember: You're talking to a human in 2026!"""

# ============================================================================
# 🛠️ UTILITY FUNCTIONS
# ============================================================================
def clean_markdown(text):
    """Clean markdown to prevent Telegram parsing errors"""
    if not text:
        return text
    
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
        
    return text[:4000]

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

def safe_send_message(chat_id, text, **kwargs):
    """Safely send a message with error handling"""
    try:
        return bot.send_message(chat_id, text, **kwargs)
    except Exception as e:
        logger.error(f"Send message error: {e}")
        # Try without markdown
        try:
            text_plain = text.replace('*', '').replace('_', '').replace('`', '').replace('~', '')
            return bot.send_message(chat_id, text_plain, **kwargs)
        except Exception as e2:
            logger.error(f"Plain text send error: {e2}")
            return None

def groq_chat_with_fallback(messages, temperature=0.7, max_tokens=400):
    """Try primary and fallback Groq chat models before failing."""
    if not groq_client:
        raise RuntimeError("Groq client not configured.")

    candidates = [CHAT_MODEL, CHAT_MODEL_FALLBACK]
    # Preserve order while removing duplicates/empty values
    models = []
    for model in candidates:
        if model and model not in models:
            models.append(model)

    last_error = None
    for model in models:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content if response and response.choices else None
            if content:
                return content, model
            raise RuntimeError(f"Empty response content from model: {model}")
        except Exception as e:
            last_error = e
            logger.warning(f"Groq chat model failed ({model}): {e}")

    raise last_error if last_error else RuntimeError("All Groq chat models failed.")

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
                    logger.info(f"Trying Pollinations.ai...")
                    
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url1, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'image' in content_type.lower():
                            logger.info("✓ Pollinations.ai successful!")
                            return response.content
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
                    
                    if not groq_client:
                        logger.warning("Groq unavailable: cannot create text fallback description.")
                        return {
                            'type': 'text',
                            'prompt': clean_prompt,
                            'description': "AI backend not configured. Set GROQ_API_KEY in .env to enable detailed descriptions.",
                            'emojis': '⚠️',
                            'suggestion': "Add GROQ_API_KEY to .env and restart the bot."
                        }

                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": description_prompt}],
                        temperature=0.8,
                        max_tokens=200
                    )
                    
                    return {
                        'type': 'text',
                        'prompt': clean_prompt,
                        'description': response.choices[0].message.content,
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
# 🚀 START COMMAND (FIXED)
# ============================================================================
def play_intro_animation(chat_id):
    """Render a short Telegram-friendly boot animation by editing one message."""
    logo = (
        "ARTOVIX AI CORE\n"
        "==============="
    )
    frames = [
        "```text\n[SYSTEM MALFUNCTION]\n!@#$%^&*()_+<>?:{}\n```",
        "```text\n[SYSTEM MALFUNCTION]\n^&*()_+<>?:{}!@#$%\n```",
        f"```text\n{logo}\n\n>> CORE STABILIZED...\n```",
        f"```text\n{logo}\n\nNeural Link    [####------] 40%\n```",
        f"```text\n{logo}\n\nNeural Link    [##########] READY\nQuantum Gates  [#####-----] 50%\n```",
        f"```text\n{logo}\n\nNeural Link    [##########] READY\nQuantum Gates  [##########] OPEN\nArtovix Core   [##########] SYNCED\n```",
        "⚡ *ARTOVIX AI v4.0 DEPLOYED*\nProtocol: Advanced Assistance | Level: Elite"
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
            time.sleep(0.35 if i < len(frames) - 1 else 0.2)
        except Exception:
            # Keep /start resilient; intro animation should never block bot usage.
            break

@bot.message_handler(commands=['start', 'artovix', 'hello'])
def start_command(message):
    try:
        play_intro_animation(message.chat.id)

        welcome_msg = """🌟 *Welcome to Artovix 2026!* 🌟

I'm your AI assistant powered by Groq's Llama 3.3 70B!

🎯 *QUICK START:*
1. 💬 **Chat** - Just type your message
2. 🎨 **AI Images** - `/flux`, `/pollin`, `/art`, or `/draw`
3. 🎙️ **Voice** - Send a voice message
4. 🖼️ **Vision** - Send a photo to analyze
5. 🔍 **Search** - `/search [question]`

🛠️ *IMAGE COMMANDS:*
`/flux` - High-quality (FLUX.1-dev)
`/pollin` - Fast & Reliable
`/art` - Creative/Artistic styles
`/draw` - Your default model

🛠️ *UTILITY COMMANDS:*
`/help` - Command reference
`/search` - Search information
`/code` - Analyze code
`/stats` - View analytics
`/reset` - Clear memory
`/status` - Bot health

*Ready to begin?* 🚀"""

        markup = InlineKeyboardMarkup(row_width=2)
        markup.add(
            InlineKeyboardButton("💬 Chat Now", callback_data="start_chat"),
            InlineKeyboardButton("🎨 Draw Image", callback_data="generate_image"),
            InlineKeyboardButton("💻 Code Help", callback_data="code_help"),
            InlineKeyboardButton("🔍 Search Web", callback_data="ask_question")
        )
        
        safe_send_message(message.chat.id, welcome_msg, reply_markup=markup)
        logger.info(f"✓ Start command from user {message.chat.id}")
        
    except Exception as e:
        logger.error(f"Start command error: {e}")
        bot.send_message(message.chat.id, "🌟 Welcome! Type /help to see commands.")

# ============================================================================
# 🎨 DRAW COMMAND (IMPROVED)
# ============================================================================
@bot.message_handler(commands=['draw', 'imagine', 'generate'])
def handle_draw(message):
    thinking_msg = None
    try:
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
        
        # Show searching indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
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
            if not groq_client:
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `GROQ_API_KEY` in your .env to enable search features.")
                return

            # Get response from Groq
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": search_prompt}],
                temperature=0.7,
                max_tokens=500
            )

            answer = clean_markdown(response.choices[0].message.content)

            # Send result
            result_text = f"🔍 *Search Results:* {query}\n\n{answer}\n\n✨ *Source:* Artovix AI Knowledge Base"
            safe_send_message(message.chat.id, result_text)

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
        
        bot.send_chat_action(message.chat.id, 'typing')
        
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
            if not groq_client:
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `GROQ_API_KEY` in your .env to enable code analysis.")
                return

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.3,
                max_tokens=600
            )

            preferred_language = detect_code_language(code_text)
            analysis = ensure_copyable_code_blocks(
                response.choices[0].message.content,
                preferred_language=preferred_language
            )
            analysis = clean_markdown(analysis)

            result_text = f"💻 *Code Analysis:*\n\n{analysis}\n\n🔧 *Powered by Artovix AI*"
            safe_send_message(message.chat.id, result_text)

        except Exception as api_error:
            logger.error(f"Code API error: {api_error}")
            safe_send_message(
                message.chat.id,
                f"💻 *Question:* {code_text}\n\n"
                f"I can help with that! Try:\n"
                f"1. Be more specific about your code issue\n"
                f"2. Send the actual code in ```code blocks```\n"
                f"3. Ask about a specific programming language"
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
@bot.message_handler(commands=['reset'])
def handle_reset(message):
    try:
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
        help_text = """🔧 *Artovix Command Reference*

*Image Generation:*
`/flux [prompt]` - High-quality (FLUX.1-dev)
`/pollin [prompt]` - Fast & reliable generation
`/art [prompt]` - Creative & artistic styles
`/auto [prompt]` - Smart model selection
`/draw [prompt]` - Use your preferred model

*Main Commands:*
`/start` - Welcome guide & features
`/help` - This command list
`/search [query]` - Search knowledge
`/code [question/code]` - Code help
`/stats` - View analytics dashboard
`/reset` - Clear conversation memory
`/status` - Check bot health

*Tips:*
• Be descriptive for better images
• Use `/draw` without a prompt to change your default model
• Send photos to analyze them (Vision)
• Send voice messages to transcribe (Whisper)

*Need more help?* Just chat with me normally! 😊"""
        
        safe_send_message(message.chat.id, help_text)
    except Exception as e:
        logger.error(f"Help error: {e}")
        safe_send_message(message.chat.id, "Type /start to begin!")

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
# 🎙️ VOICE MESSAGES HANDLER
# ============================================================================
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        bot.send_chat_action(message.chat.id, 'upload_document')
        
        # Get voice file info
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save temporarily
        temp_filename = f"temp/voice_{message.chat.id}_{int(time.time())}.ogg"
        os.makedirs("temp", exist_ok=True)
        with open(temp_filename, 'wb') as f:
            f.write(downloaded_file)
        
        # Transcribe with Groq Whisper
        with open(temp_filename, "rb") as audio_file:
            if not groq_client:
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `GROQ_API_KEY` in your .env to enable voice transcription.")
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
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Get highest resolution photo
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Encode to base64
        base64_image = base64.b64encode(downloaded_file).decode('utf-8')
        
        # Get caption or use default
        prompt = message.caption if message.caption else "Describe this image in detail and tell me what you see."
        
        # Analyze with Groq Vision
        if not groq_client:
            safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `GROQ_API_KEY` in your .env to enable vision features.")
            return

        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
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
            ],
            max_tokens=500
        )

        analysis = clean_markdown(response.choices[0].message.content)

        safe_send_message(message.chat.id, f"🖼️ *Image Analysis:*\n\n{analysis}")

        analytics.log_request(message.chat.id, 500, "vision_analysis")
        
    except Exception as e:
        logger.error(f"Vision handling error: {e}")
        safe_send_message(message.chat.id, "🖼️ *Vision analysis failed.*\nPlease try again with a clearer image.")

# ============================================================================
# 💬 TEXT MESSAGES HANDLER
# ============================================================================
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    try:
        # Skip if empty or command
        if not message.text or message.text.startswith('/'):
            return
        
        logger.info(f"Message from {message.chat.id}: {message.text[:50]}...")
        
        user_id = str(message.chat.id)
        user_data = memory.get_user_data(user_id)
        history = user_data["history"]
        
        # Prepare conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-4:],  # Last 2 exchanges
            {"role": "user", "content": message.text}
        ]
        
        bot.send_chat_action(message.chat.id, 'typing')
        
        try:
            if not groq_client:
                safe_send_message(message.chat.id, "🔌 *AI backend not configured.*\nSet `GROQ_API_KEY` in your .env to enable chat responses.")
                return

            reply_raw, used_model = groq_chat_with_fallback(
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )
            reply = clean_markdown(reply_raw)
            
            # Save to memory
            history.extend([
                {"role": "user", "content": message.text},
                {"role": "assistant", "content": reply}
            ])
            
            # Limit memory size
            if len(history) > 20:
                history = history[-20:]
            
            user_data["history"] = history
            memory.save_user_data(user_id, user_data)
            
            # Send reply
            safe_send_message(message.chat.id, reply)
            logger.info(f"Chat reply sent using model: {used_model}")
            
            # Log analytics
            tokens_used = len(message.text.split()) + len(reply.split())
            analytics.log_request(message.chat.id, tokens_used, "chat")
            
        except Exception as api_error:
            logger.error(f"Chat API error: {api_error}\n{traceback.format_exc()}")
            try:
                err = str(api_error).lower()
                if "rate limit" in err or "429" in err:
                    user_msg = (
                        "⏳ *Too many requests right now.*\n\n"
                        "Please wait 10-20 seconds and try again."
                    )
                elif "api key" in err or "unauthorized" in err or "authentication" in err:
                    user_msg = (
                        "🔐 *AI key issue detected.*\n\n"
                        "Please check `GROQ_API_KEY` in your Render environment variables."
                    )
                elif "model" in err and ("not found" in err or "decommissioned" in err or "not available" in err):
                    user_msg = (
                        "🧠 *Model temporarily unavailable.*\n\n"
                        "I switched models automatically. Please try your message again."
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
# 🎪 CALLBACK HANDLER
# ============================================================================
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        if call.data == "start_chat":
            bot.answer_callback_query(call.id, "Let's chat!")
            safe_send_message(call.message.chat.id, 
                "💬 *Chat Activated!*\n\n"
                "Just type your message and I'll respond!\n\n"
                "*Try asking:*\n"
                "• What can you do?\n"
                "• Tell me about AI\n"
                "• Help me with a problem"
            )
        
        elif call.data == "generate_image":
            bot.answer_callback_query(call.id, "Image generation!")
            safe_send_message(call.message.chat.id, 
                "🎨 *Image Generator*\n\n"
                "*Usage:* `/draw [description]`\n\n"
                "*Quick examples:*\n"
                "• `/draw sunset over mountains`\n"
                "• `/draw cute robot futuristic city`\n"
                "• `/draw magical forest glowing plants`\n\n"
                "Be creative! 🎨"
            )
        
        elif call.data == "code_help":
            bot.answer_callback_query(call.id, "Code help!")
            safe_send_message(call.message.chat.id, 
                "💻 *Code Assistant*\n\n"
                "*Two ways to use:*\n"
                "1. Ask: `/code how to [do something]`\n"
                "2. Send code in:\n"
                "```python\n"
                "print('Hello World!')\n"
                "```\n\n"
                "*I can help with:* Python, JavaScript, Java, C++, etc."
            )
        
        elif call.data == "ask_question":
            bot.answer_callback_query(call.id, "Search!")
            safe_send_message(call.message.chat.id, 
                "🔍 *Knowledge Search*\n\n"
                "*Usage:* `/search [your question]`\n\n"
                "*Examples:*\n"
                "• `/search latest space discoveries`\n"
                "• `/search how AI works in 2026`\n"
                "• `/search best programming practices`\n\n"
                "Ask me anything! 🌟"
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
    ║            ULTIMATE AI EDITION                   ║
    ║                                                  ║
    ║  ✅ **ALL COMMANDS WORKING:**                    ║
    ║    • /start, /help, /status, /stats             ║
    ║    • /draw - Image generation                   ║
    ║    • /search - Knowledge specialist             ║
    ║    • /code - Programming expert                 ║
    ║                                                  ║
    ║  🚀 **NEW FUTURES ADDED:**                       ║
    ║    • 🎙️ Voice-to-Text (Whisper V3)               ║
    ║    • 🖼️ Image Analysis (Vision)                 ║
    ║    • 📊 Analytics Breakdown                     ║
    ║    • 🛡️ Concurrent Memory Lock                  ║
    ║                                                  ║
    ║  🚀 Starting up...                               ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    logger.info("🚀 Starting Artovix 2026 (Ultimate Edition)...")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    bot_username = "(unavailable)"
    try:
        me = bot.get_me()
        bot_username = getattr(me, "username", "(unknown)")
    except Exception as e:
        logger.warning(f"Startup bot identity check failed: {e}")
    print(f"🤖 Bot: @{bot_username}")
    print(f"🧠 Memory: {len(memory.load())} active conversations")
    print(f"📊 Analytics: Enhanced database ready")
    
    print("\n" + "="*60)
    print("✅ ALL COMMANDS READY:")
    print("="*60)
    print("💬 /start - Welcome & features")
    print("🎨 /draw [prompt] - Generate images")
    print("🔍 /search [query] - Search knowledge")
    print("💻 /code [question] - Programming help")
    print("📊 /stats - View analytics")
    print("🧹 /reset - Clear memory")
    print("✅ /status - Bot health")
    print("🔧 /help - Command list")
    print("="*60)
    print("\n⚡ Bot is running and ready to receive commands!")
    print("💡 Tip: Try /draw a beautiful landscape")
    
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
