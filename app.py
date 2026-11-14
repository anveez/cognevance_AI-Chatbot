import sys
import time
from collections import Counter
from datetime import datetime

sys.path.append("src")

import pandas as pd
import streamlit as st
from chatbot import Chatbot

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="TechBuddy AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

INTENT_META = {
    "greeting": ("👋", "Greeting"),
    "goodbye": ("🚪", "Goodbye"),
    "thanks": ("🙏", "Thanks"),
    "name": ("🪪", "Identity"),
    "help": ("🛟", "Help"),
    "hours": ("🕘", "Hours"),
    "contact": ("📞", "Contact"),
    "pricing": ("💳", "Pricing"),
    "refund": ("↩️", "Refund"),
    "joke": ("😄", "Joke"),
    "fallback": ("❓", "Unknown"),
}

QUICK_REPLIES = [
    ("💳 Pricing", "How much does it cost?"),
    ("🕘 Opening hours", "What are your opening hours?"),
    ("↩️ Refund policy", "What is your refund policy?"),
    ("📞 Contact", "How can I contact you?"),
    ("😄 Tell a joke", "Tell me a joke"),
]

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 6rem; max-width: 900px; }

/* animated gradient background */
.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(124,92,255,.18), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(0,200,255,.12), transparent 55%),
        #0b0e17;
}

/* hero header */
.hero {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 22px; margin-bottom: 14px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(124,92,255,.25), rgba(0,200,255,.12));
    border: 1px solid rgba(255,255,255,.08);
    backdrop-filter: blur(10px);
}
.hero .logo {
    width: 52px; height: 52px; border-radius: 14px;
    display: grid; place-items: center; font-size: 28px;
    background: linear-gradient(135deg, #7c5cff, #00c8ff);
    box-shadow: 0 6px 22px rgba(124,92,255,.45);
}
.hero h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: #fff; }
.hero p  { margin: 2px 0 0; font-size: .85rem; color: #aab2c8; }
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #2ee59d; margin-right: 6px;
    box-shadow: 0 0 0 0 rgba(46,229,157,.7); animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(46,229,157,.6); }
    70% { box-shadow: 0 0 0 9px rgba(46,229,157,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,229,157,0); }
}

/* chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 18px; padding: 14px 18px; margin-bottom: 10px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    animation: fadeUp .35s ease;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(124,92,255,.35), rgba(91,140,255,.25));
    border-color: rgba(124,92,255,.35);
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px);} to { opacity: 1; transform: none;} }

/* meta row under bot messages */
.meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 8px; }
.badge {
    padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600;
    background: rgba(124,92,255,.18); color: #c9bdff; border: 1px solid rgba(124,92,255,.35);
}
.badge.warn { background: rgba(255,170,60,.15); color: #ffc980; border-color: rgba(255,170,60,.35); }
.time { font-size: .72rem; color: #7d869e; }
.conf-wrap { width: 110px; height: 6px; border-radius: 6px; background: rgba(255,255,255,.1); overflow: hidden; }
.conf-bar  { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7c5cff, #00c8ff); }

/* quick reply chips */
div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100%; border-radius: 999px; padding: 6px 10px; font-size: .82rem;
    background: rgba(255,255,255,.05); color: #dfe3f0;
    border: 1px solid rgba(255,255,255,.12); transition: all .2s ease;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(124,92,255,.25); border-color: #7c5cff; transform: translateY(-2px);
}

/* chat input */
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid rgba(124,92,255,.4); }

/* sidebar */
section[data-testid="stSidebar"] {
    background: rgba(14,18,30,.95); border-right: 1px solid rgba(255,255,255,.06);
}
.stat-card {
    padding: 12px 14px; border-radius: 14px; text-align: center;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
}
.stat-card .num { font-size: 1.5rem; font-weight: 700; color: #fff; }
.stat-card .lbl { font-size: .72rem; color: #8f98b3; text-transform: uppercase; letter-spacing: .06em; }

/* ---------- FORCE READABLE COLORS (works in light or dark Streamlit theme) ---------- */
:root { color-scheme: dark; }
.stApp { color: #e6e8f0; }

/* main text */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 { color: #e6e8f0 !important; }
.hero h1 { color: #ffffff !important; }
.hero p  { color: #aab2c8 !important; }

/* sidebar text, labels, slider value, captions */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #e6e8f0 !important; }
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { color: #aab2c8 !important; }

/* sidebar buttons (Export / Clear) */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(255,255,255,.06) !important;
    color: #e6e8f0 !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 12px;
}
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: rgba(124,92,255,.25) !important; border-color: #7c5cff !important;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stDownloadButton > button p { color: #e6e8f0 !important; }

/* quick-reply chips text */
.stApp .stButton > button p { color: #dfe3f0 !important; }

/* bottom bar + chat input (was white) */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div { background: #141a2b !important; }
[data-testid="stChatInput"] textarea {
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8f98b3 !important; -webkit-text-fill-color: #8f98b3 !important; }
[data-testid="stChatInput"] button { background: #7c5cff !important; }
[data-testid="stChatInput"] button svg { color: #ffffff !important; fill: #ffffff !important; }

/* spinner + chart text */
.stApp [data-testid="stSpinner"] p { color: #aab2c8 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------
@st.cache_resource
def load_bot():
    return Chatbot()


bot = load_bot()

WELCOME = {
    "role": "assistant",
    "content": "Hi! I'm **TechBuddy** 👋 Ask me about pricing, opening hours, refunds or contact details, "
               "or pick a quick option below.",
    "time": datetime.now().strftime("%H:%M"),
    "intent": "greeting",
    "conf": 1.0,
}

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME]
if "pending" not in st.session_state:
    st.session_state.pending = None

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05,
                          help="Below this confidence the bot uses the fallback reply.")
    typing_effect = st.toggle("Typing animation", value=True)
    show_meta = st.toggle("Show intent and confidence", value=True)
    bot.threshold = threshold

    st.divider()
    st.markdown("### 📊 Session stats")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    bot_msgs = [m for m in st.session_state.messages if m["role"] == "assistant" and m is not WELCOME]
    confs = [m["conf"] for m in bot_msgs]
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat-card"><div class="num">{len(user_msgs)}</div>'
                f'<div class="lbl">Messages</div></div>', unsafe_allow_html=True)
    avg = f"{(sum(confs) / len(confs)) * 100:.0f}%" if confs else "–"
    c2.markdown(f'<div class="stat-card"><div class="num">{avg}</div>'
                f'<div class="lbl">Avg confidence</div></div>', unsafe_allow_html=True)

    counts = Counter(m["intent"] for m in bot_msgs)
    if counts:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.caption("Detected intents")
        df = pd.DataFrame(
            {"Intent": [INTENT_META.get(k, ("", k))[1] for k in counts], "Count": list(counts.values())}
        ).set_index("Intent")
        st.bar_chart(df, height=180, color="#7c5cff")

    st.divider()
    chat_text = "\n".join(
        f'[{m["time"]}] {"You" if m["role"] == "user" else "TechBuddy"}: {m["content"]}'
        for m in st.session_state.messages
    )
    st.download_button("⬇️ Export chat", chat_text, file_name="chat_history.txt",
                       use_container_width=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = [WELCOME]
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption("Built with NLTK · scikit-learn · Streamlit")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="logo">🤖</div>
  <div>
    <h1>TechBuddy AI Assistant</h1>
    <p><span class="status-dot"></span>Online · NLP intent-based chatbot</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def meta_html(msg):
    emoji, label = INTENT_META.get(msg["intent"], ("💬", msg["intent"]))
    cls = "badge warn" if msg["intent"] == "fallback" else "badge"
    pct = int(msg["conf"] * 100)
    intent_part = (
        f'<span class="{cls}">{emoji} {label}</span>'
        f'<div class="conf-wrap"><div class="conf-bar" style="width:{pct}%"></div></div>'
        f'<span class="time">{pct}%</span>'
    ) if show_meta else ""
    return f'<div class="meta">{intent_part}<span class="time">{msg["time"]}</span></div>'


def render(msg):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.markdown(meta_html(msg), unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="meta"><span class="time">{msg["time"]}</span></div>',
                        unsafe_allow_html=True)


# ------------------------------------------------------------------
# Render history
# ------------------------------------------------------------------
for m in st.session_state.messages:
    render(m)

# ------------------------------------------------------------------
# Quick replies (only shown at the start of a conversation)
# ------------------------------------------------------------------
if len(st.session_state.messages) <= 1:
    st.markdown("<p style='color:#8f98b3;font-size:.85rem;margin:6px 0 4px'>Try asking:</p>",
                unsafe_allow_html=True)
    cols = st.columns(len(QUICK_REPLIES))
    for col, (label, query) in zip(cols, QUICK_REPLIES):
        if col.button(label, key=f"qr_{label}"):
            st.session_state.pending = query
            st.rerun()

# ------------------------------------------------------------------
# Handle new input
# ------------------------------------------------------------------
typed = st.chat_input("Type your message...")
prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt:
    now = datetime.now().strftime("%H:%M")
    user_msg = {"role": "user", "content": prompt, "time": now}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("TechBuddy is thinking..."):
            time.sleep(0.4)
            answer, tag, conf = bot.reply(prompt)

        if typing_effect:
            holder = st.empty()
            shown = ""
            for word in answer.split(" "):
                shown += word + " "
                holder.markdown(shown + "▌")
                time.sleep(0.035)
            holder.markdown(shown)
        else:
            st.markdown(answer)

        bot_msg = {
            "role": "assistant", "content": answer,
            "time": datetime.now().strftime("%H:%M"),
            "intent": tag, "conf": conf,
        }
        st.markdown(meta_html(bot_msg), unsafe_allow_html=True)

    st.session_state.messages.append(bot_msg)
    st.rerun()