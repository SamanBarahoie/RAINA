# file: chatbot_app_streamlit_final_retry.py
import uuid
import json
import logging
import requests
import streamlit as st
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="راینا", page_icon="🤖", layout="wide")

# ==============================
# API URL Configuration
# ==============================
API_URL = os.environ.get("FASTAPI_URL", "http://fastapi:80") + "/api/chat"

# -------------------------------------------------------------------
# Wait for FastAPI to be ready
# -------------------------------------------------------------------
MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds

for attempt in range(MAX_RETRIES):
    try:
        resp = requests.get(API_URL.replace("/api/chat", "/"), timeout=5)
        if resp.status_code == 200:
            logger.info("FastAPI is ready!")
            break
    except requests.exceptions.RequestException:
        logger.info(f"FastAPI not ready, retrying ({attempt + 1}/{MAX_RETRIES})...")
        time.sleep(RETRY_DELAY)
else:
    logger.warning("FastAPI did not respond after multiple retries. Requests may fail.")

# -------------------------------------------------------------------
# Generate a random USER_ID only once per Streamlit session
# -------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex}"

# -------------------------------------------------------------------
# CSS Styling
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');
body, .stApp { direction: rtl; font-family: 'Vazirmatn', sans-serif; background: radial-gradient(circle at 30% 20%, #0a0a12, #101018, #0f0f16); color: #e5e5e5; overflow-x: hidden; }
.main .block-container { background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(20px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); padding: 2.5rem 3rem; margin-top: 2rem; border: 1px solid rgba(255,255,255,0.08); }
h1 { text-align: center; color: #b5c9ff; font-weight: 700; text-shadow: 0 0 12px rgba(120,150,255,0.8); letter-spacing: 1px; }
.intro-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; z-index: 999; animation: fadeIn 1s ease; }
.intro-container img { width: 160px; height: 160px; border-radius: 50%; box-shadow: 0 0 35px rgba(120,150,255,0.7); margin-bottom: 20px; }
.typing { text-align: center; color: #9bb8ff; font-size: 1.2rem; border-right: 3px solid #89a6ff; white-space: nowrap; overflow: hidden; animation: typing 3.5s steps(40, end), blink 0.7s step-end infinite alternate; }
@keyframes typing { from { width: 0 } to { width: 100% } }
@keyframes blink { from, to { border-color: transparent } 50% { border-color: #89a6ff; } }
@keyframes fadeIn { from {opacity: 0; transform: translate(-50%, -45%);} to {opacity: 1; transform: translate(-50%, -50%);} }
.stChatMessage { direction: rtl; text-align: right; border-radius: 18px; margin-bottom: 1rem; padding: 1rem 1.25rem; line-height: 1.6; animation: fadeInMsg 0.4s ease; color: #e8e8e8; }
@keyframes fadeInMsg { from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: translateY(0);} }
.stChatMessage[data-testid="chat-message-container-user"] { background: linear-gradient(135deg, rgba(32, 90, 250, 0.3), rgba(80, 120, 255, 0.25)); border: 1px solid rgba(100,150,255,0.25); }
.stChatMessage[data-testid="chat-message-container-assistant"] { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); }
[data-testid="chat-input"] { direction: rtl; background: rgba(255,255,255,0.08); color: #ffffff; border-radius: 14px; border: 1px solid rgba(255,255,255,0.15); padding: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Page Title
# -------------------------------------------------------------------
st.title("راینا دستیار هوشمند آیین‌نامه‌های دانشگاه")

# -------------------------------------------------------------------
# API Call Wrapper
# -------------------------------------------------------------------
def get_chat_response(user_id: str, message: str, session_id: str | None):
    """Send message to backend API via POST and get assistant response."""
    payload = {"user_id": user_id, "message": message, "session_id": session_id}
    headers = {"Content-Type": "application/json"}

    try:
        res = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("reply", "پاسخی دریافت نشد."), data.get("session_id")
    except Exception as exc:
        logger.error(f"Error connecting to API: {exc}")
        return f"⚠️ خطا در ارتباط با سرور: {exc}", session_id

# -------------------------------------------------------------------
# Session State Init
# -------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("intro_shown", False)

# -------------------------------------------------------------------
# Intro Animation (Shown Only Once)
# -------------------------------------------------------------------
if not st.session_state.intro_shown and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="intro-container">
        <img src="https://i.gifer.com/XDZT.gif" alt="AI Animation">
        <div class="typing">👋 سلام... من راینا هستم — دستیار هوشمند شما 💫</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.intro_shown = True

# -------------------------------------------------------------------
# Display Chat Messages
# -------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------------
# User Input Handling
# -------------------------------------------------------------------
if prompt := st.chat_input("پیام خود را بنویسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            reply, new_session = get_chat_response(
                st.session_state.user_id,
                prompt,
                st.session_state.session_id
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    if new_session:
        st.session_state.session_id = new_session
