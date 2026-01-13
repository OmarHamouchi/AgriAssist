# app.py
import time
import html
import re
import streamlit as st
import streamlit.components.v1 as components
from agri_backend import SmartAgriAgent

# =========================
# Page config
# =========================
st.set_page_config(
    layout="wide",
    page_title="AgriAssist AI",
    page_icon="🌾",
    initial_sidebar_state="collapsed",
)

# =========================
# Global CSS (remove white bar + page padding)
# =========================
st.markdown(
    """
<style>
/* Remove Streamlit top white bar */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

/* Remove reserved spaces */
[data-testid="stAppViewContainer"]{ padding-top: 0px !important; }
[data-testid="stMainBlockContainer"]{
  padding-top: 0px !important;
  padding-bottom: 14px !important;
  max-width: 1400px !important;
}

/* Hide Streamlit chrome */
footer{display:none;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stSidebarNav"]{display:none;}
section[data-testid="stSidebar"]{display:none;}

/* Soft agriculture background */
.stApp{
  background: radial-gradient(1200px 600px at 20% 10%, #EAF3E4, #F4F7F2);
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))

def esc(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")

def classify_log(line: str) -> str:
    s = line or ""
    if any(x in s for x in ["❌", "Error", "Erreur", "ERREUR", "Connection Error"]):
        return "l-err"
    if any(x in s for x in ["⚠️", "WARNING", "Pas de clé", "introuvable"]):
        return "l-warn"
    if any(x in s for x in ["✅", "prêt", "COMPLETE", "Complete"]):
        return "l-ok"
    return "l-sys"

# =========================
# Components HTML (NO markdown parsing => no HTML shown as text)
# =========================
def phone_html(messages, typing_text: str | None = None) -> str:
    bubbles = []
    for m in messages:
        role = m.get("role", "assistant")
        raw = m.get("content", "")
        safe = esc(raw)

        if role == "user":
            bubbles.append(
                f"""
                <div class="msg user">
                  <div class="bubble user">{safe}</div>
                  <div class="avatar">🧑‍🌾</div>
                </div>
                """
            )
        else:
            dir_style = "direction: rtl; text-align: right;" if is_arabic(raw) else ""
            bubbles.append(
                f"""
                <div class="msg assistant">
                  <div class="avatar">🤖</div>
                  <div class="bubble assistant" style="{dir_style}">{safe}</div>
                </div>
                """
            )

    if typing_text is not None:
        dir_style = "direction: rtl; text-align: right;" if is_arabic(typing_text) else ""
        bubbles.append(
            f"""
            <div class="msg assistant">
              <div class="avatar">🤖</div>
              <div class="bubble assistant" style="{dir_style}">
                {esc(typing_text)} <span class="caret">▌</span>
              </div>
            </div>
            """
        )

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body {{ height: 100%; }}
body {{
  margin: 0;
  background: transparent;
  font-family: "Poppins", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}
:root {{
  --leaf:#2E7D32;
  --leaf2:#1B5E20;
  --sun:#F6C453;
  --text:#0F172A;
  --muted:#64748B;
}}
.panel {{
  height: 100%;
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(15, 23, 42, .10);
  border: 1px solid rgba(27, 94, 32, .14);
  background: linear-gradient(180deg, #FFFFFF 0%, #FAFFFA 70%, #FFFFFF 100%);
}}
.topbar {{
  background: linear-gradient(90deg, var(--leaf2) 0%, var(--leaf) 55%, #2f8f37 100%);
  color: white;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 700;
}}
.brand {{
  display: flex;
  gap: 10px;
  align-items: center;
}}
.pill {{
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.18);
}}
.dot {{
  width: 10px; height: 10px; border-radius: 50%;
  background: #34D399;
  box-shadow: 0 0 0 0 rgba(52,211,153,.55);
  animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
  0%{{ box-shadow:0 0 0 0 rgba(52,211,153,.55); }}
  70%{{ box-shadow:0 0 0 10px rgba(52,211,153,0); }}
  100%{{ box-shadow:0 0 0 0 rgba(52,211,153,0); }}
}}

.body {{
  height: calc(100% - 56px - 40px);
  padding: 16px 14px 8px 14px;
  overflow-y: auto;
  background:
    radial-gradient(900px 220px at 10% 0%, rgba(46,125,50,.07), transparent 60%),
    radial-gradient(600px 240px at 90% 10%, rgba(246,196,83,.08), transparent 62%),
    linear-gradient(180deg, #FFFFFF 0%, #FBFFFB 100%);
}}
.body::-webkit-scrollbar{{ width:8px; }}
.body::-webkit-scrollbar-thumb{{ background: rgba(27,94,32,.22); border-radius:999px; }}

.msg {{
  display: flex;
  gap: 10px;
  margin: 10px 0;
}}
.msg.user {{ justify-content: flex-end; }}
.msg.assistant {{ justify-content: flex-start; }}

.avatar {{
  width: 34px; height: 34px;
  display: flex; align-items:center; justify-content:center;
  border-radius: 12px;
  background: rgba(46,125,50,.08);
  border: 1px solid rgba(46,125,50,.12);
  font-size: 22px;
}}

.bubble {{
  max-width: 76%;
  padding: 12px 14px;
  border-radius: 18px;
  line-height: 1.55;
  font-size: 14.2px;
  font-weight: 500;
  box-shadow: 0 10px 22px rgba(2, 6, 23, .06);
  border: 1px solid rgba(15,23,42,.06);
  word-wrap: break-word;
}}
.bubble.user {{
  color: #0B1220;
  background: linear-gradient(135deg, rgba(246,196,83,.92), rgba(246,196,83,.72));
  border-bottom-right-radius: 6px;
}}
.bubble.assistant {{
  background: #F3F7F2;
  color: var(--text);
  border-left: 4px solid rgba(46,125,50,.7);
  border-bottom-left-radius: 6px;
}}

.hint {{
  height: 40px;
  padding: 10px 14px;
  border-top: 1px solid rgba(15,23,42,.06);
  background: linear-gradient(180deg, rgba(255,255,255,.65) 0%, #FFFFFF 60%);
  font-size: 12.5px;
  color: var(--muted);
  display:flex;
  align-items:center;
}}

.caret {{
  color: rgba(46,125,50,.9);
  font-weight: 800;
  animation: blink 1s infinite;
}}
@keyframes blink {{ 0%,50%{{opacity:1;}} 51%,100%{{opacity:0;}} }}
</style>
</head>
<body>
  <div class="panel">
    <div class="topbar">
      <div class="brand">
        <span class="dot"></span>
        <span>AgriAssist</span>
        <span class="pill">Arabic • RAG</span>
      </div>
      <span class="pill">🌾 Agriculture</span>
    </div>

    <div class="body">
      {''.join(bubbles)}
    </div>

    <div class="hint">
      💡 اكتب سؤالك أسفل… مثال: "كيف أتعامل مع المنّ في الطماطم؟"
    </div>
  </div>
</body>
</html>
"""

def terminal_html(logs, tail: int = 140) -> str:
    view = logs[-tail:] if len(logs) > tail else logs
    lines = []
    for line in view:
        cls = classify_log(line)
        lines.append(f'<div class="line {cls}">{esc(line)}</div>')

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap');
html, body {{ height: 100%; }}
body {{
  margin: 0;
  background: transparent;
  font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}
:root {{
  --term:#0B1220;
  --term2:#0F1A30;
  --termText:#A7F3D0;
  --termMuted:#93C5FD;
  --termWarn:#FBBF24;
  --termErr:#FB7185;
  --termOk:#34D399;
}}
.panel {{
  height: 100%;
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(15, 23, 42, .10);
  border: 1px solid rgba(147,197,253,.14);
  background: linear-gradient(180deg, var(--term2) 0%, var(--term) 100%);
}}
.topbar {{
  padding: 14px 16px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  border-bottom: 1px solid rgba(147,197,253,.18);
  background: linear-gradient(90deg, rgba(147,197,253,.08), rgba(52,211,153,.06));
}}
.title {{
  color: var(--termMuted);
  font-weight: 600;
  font-size: 13px;
}}
.title b {{ color: var(--termText); font-weight: 700; }}
.badge {{
  display:flex; align-items:center; gap:10px;
  color: rgba(167,243,208,.92);
  font-size: 12.5px;
}}
.dot {{
  width:9px;height:9px;border-radius:50%;
  background: var(--termOk);
  box-shadow: 0 0 0 0 rgba(52,211,153,.55);
  animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
  0%{{ box-shadow:0 0 0 0 rgba(52,211,153,.55); }}
  70%{{ box-shadow:0 0 0 10px rgba(52,211,153,0); }}
  100%{{ box-shadow:0 0 0 0 rgba(52,211,153,0); }}
}}
.body {{
  height: calc(100% - 56px);
  padding: 14px 14px 18px 14px;
  overflow-y: auto;
  font-size: 12.8px;
  color: var(--termText);
  line-height: 1.6;
}}
.body::-webkit-scrollbar{{ width:10px; }}
.body::-webkit-scrollbar-thumb{{ background: rgba(147,197,253,.22); border-radius:999px; }}

.line {{
  padding: 7px 10px;
  border-radius: 10px;
  margin: 6px 0;
  background: rgba(148,163,184,.06);
  border: 1px solid rgba(148,163,184,.08);
}}
.l-ok{{ color: var(--termOk); border-left: 3px solid var(--termOk); }}
.l-warn{{ color: var(--termWarn); border-left: 3px solid var(--termWarn); }}
.l-err{{ color: var(--termErr); border-left: 3px solid var(--termErr); }}
.l-sys{{ color: rgba(167,243,208,.95); border-left: 3px solid rgba(147,197,253,.65); }}

.cursor {{
  margin-top: 10px;
  color: rgba(167,243,208,.75);
}}
.cursor span{{ animation: blink 1s infinite; }}
@keyframes blink {{ 0%,50%{{opacity:1;}} 51%,100%{{opacity:0;}} }}
</style>
</head>
<body>
  <div class="panel">
    <div class="topbar">
      <div class="title">root@agri-server:<b>~#</b> <span style="color:rgba(147,197,253,.95)">agri_trace</span></div>
      <div class="badge">LIVE TRACE <span class="dot"></span></div>
    </div>

    <div class="body">
      {''.join(lines)}
      <div class="cursor">&gt;&gt; <span>_</span></div>
    </div>
  </div>
</body>
</html>
"""

# =========================
# Backend
# =========================
@st.cache_resource
def get_agent():
    return SmartAgriAgent()

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Erreur Backend: {e}")
    st.stop()

# =========================
# Session state
# =========================
DEFAULT_GREETING = {"role": "assistant", "content": "مرحباً! 👋 أنا AgriAssist 🌾\nكيف يمكنني مساعدتك اليوم؟"}

if "messages" not in st.session_state:
    st.session_state.messages = [DEFAULT_GREETING]

if "logs" not in st.session_state:
    st.session_state.logs = getattr(agent, "init_logs", []).copy()

# =========================
# Reset button
# =========================
_, btn_col = st.columns([0.78, 0.22])
with btn_col:
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.messages = [DEFAULT_GREETING]
        st.session_state.logs = getattr(agent, "init_logs", []).copy()
        st.rerun()

# =========================
# Layout
# =========================
col_phone, col_term = st.columns([0.46, 0.54], gap="large")

PHONE_H = 780
TERM_H  = 780

with col_phone:
    phone_slot = st.empty()

with col_term:
    term_slot = st.empty()

def draw(phone_messages, term_logs, typing=None):
    with col_phone:
        phone_slot.empty()
        components.html(phone_html(phone_messages, typing_text=typing), height=PHONE_H, scrolling=False)

    with col_term:
        term_slot.empty()
        components.html(terminal_html(term_logs), height=TERM_H, scrolling=False)

# Initial render
draw(st.session_state.messages, st.session_state.logs)

# =========================
# Input
# =========================
with col_phone:
    prompt = st.chat_input("اكتب سؤالك هنا...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    draw(st.session_state.messages, st.session_state.logs, typing="⏳ جاري التفكير...")

    # Backend call
    response, exec_logs = agent.process_query(prompt)

    # Animate terminal logs
    base = st.session_state.logs.copy()
    for i in range(len(exec_logs)):
        draw(st.session_state.messages, base + exec_logs[: i + 1], typing="⏳ جاري التفكير...")
        time.sleep(0.03)

    st.session_state.logs.extend(exec_logs)
    st.session_state.logs.append("─" * 52)

    # Animate assistant response
    words = (response or "").split()
    typed = ""
    for w in words:
        typed += w + " "
        temp_msgs = st.session_state.messages + [{"role": "assistant", "content": typed.strip()}]
        draw(temp_msgs, st.session_state.logs)
        time.sleep(0.02)

    # Persist assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    draw(st.session_state.messages, st.session_state.logs)

    st.rerun()
