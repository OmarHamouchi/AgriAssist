# app.py
import streamlit as st
import time
from agri_backend import SmartAgriAgent

# 1. Config Page (Wide mode indispensable)
st.set_page_config(layout="wide", page_title="AgriAssist AI", page_icon="🌾")

# 2. CSS AVANCÉ (Modern UI + RTL Arabic Fix)
st.markdown("""
<style>
    /* Fond global propre */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* --- STYLE CHATBOT (Mobile View) --- */
    /* On force la direction RTL pour l'arabe */
    .stChatMessage {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Bulles de chat */
    .stChatMessage .stMarkdown {
        text-align: right !important;
    }

    /* Input bar fix */
    .stChatInput textarea {
        direction: rtl; 
        text-align: right;
    }

    /* Container du Chat style "App" */
    div[data-testid="column"]:nth-of-type(1) {
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        height: 85vh;
        overflow-y: auto;
    }

    /* En-tête fictif téléphone */
    .mobile-header {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        color: white;
        padding: 15px;
        border-radius: 15px 15px 0 0;
        text-align: center;
        margin: -20px -20px 20px -20px; /* Pour coller aux bords */
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* --- STYLE TERMINAL (System View) --- */
    div[data-testid="column"]:nth-of-type(2) {
        background-color: #0f0f0f;
        border-radius: 10px;
        padding: 0;
        border: 1px solid #333;
        height: 85vh;
        overflow: hidden; /* Scroll géré par l'intérieur */
    }

    .terminal-header {
        background-color: #1a1a1a;
        color: #ddd;
        padding: 10px;
        font-family: monospace;
        font-size: 0.9rem;
        border-bottom: 1px solid #333;
        display: flex;
        justify-content: space-between;
    }

    .terminal-body {
        padding: 15px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        color: #00ff41; /* Vert Matrix */
        height: 100%;
        overflow-y: auto;
        line-height: 1.4;
    }

    .log-line { margin-bottom: 6px; border-bottom: 1px dashed #222; padding-bottom: 2px;}
    .log-error { color: #ff5555; }
    .log-info { color: #50fa7b; }
    .log-warn { color: #f1fa8c; }
    .log-system { color: #8be9fd; }

</style>
""", unsafe_allow_html=True)

# 3. LOGIQUE APPLICATION
@st.cache_resource
def get_agent():
    return SmartAgriAgent()

try:
    agent = get_agent()
    # Récupère logs initiaux
    if "init_logs" not in st.session_state:
        st.session_state.init_logs = agent.init_logs
except Exception as e:
    st.error(f"Erreur Backend: {e}")
    st.stop()

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً! أنا AgriAssist. كيف يمكنني مساعدتك في مزرعتك اليوم؟"}
    ]
if "logs" not in st.session_state:
    st.session_state.logs = st.session_state.init_logs

# 4. LAYOUT (2 Colonnes : 40% Mobile / 60% Terminal)
col_mobile, col_terminal = st.columns([4, 6])

# --- COLONNE GAUCHE : MOBILE APP ---
with col_mobile:
    st.markdown('<div class="mobile-header">📱 AgriAssist App</div>', unsafe_allow_html=True)
    
    # Zone de Chat
    # On utilise un container avec une hauteur fixe pour simuler l'écran
    chat_container = st.container(height=550) 
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # Input (Automatiquement en bas)
    if prompt := st.chat_input("أكتب سؤالك هنا... (Ecrivez votre question)"):
        # Affichage User
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.markdown(prompt)

        # Traitement Backend
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("...جاري التفكير (Thinking)...")
                
                # APPEL BACKEND
                response, exec_logs = agent.process_query(prompt)
                
                # Mise à jour des logs terminal
                st.session_state.logs.extend(exec_logs)
                st.session_state.logs.append("--------------------------------------------------")
                
                # Affichage réponse progressive
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.04)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Refresh pour mettre à jour le terminal à droite

# --- COLONNE DROITE : TERMINAL ---
with col_terminal:
    st.markdown("""
    <div class="terminal-header">
        <span>root@agri-server:~# ./run_pipeline.sh</span>
        <span>🔴 LIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage des Logs
    log_html = '<div class="terminal-body">'
    for log in st.session_state.logs:
        # Coloriage syntaxique simple
        css_class = "log-info"
        if "❌" in log or "Error" in log: css_class = "log-error"
        elif "⚠️" in log: css_class = "log-warn"
        elif "🔍" in log or "Routing" in log: css_class = "log-system"
        
        log_html += f'<div class="log-line {css_class}">{log}</div>'
    
    log_html += '<div class="log-line">> _</div></div>'
    st.markdown(log_html, unsafe_allow_html=True)