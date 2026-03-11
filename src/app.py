"""
Interactive Streamlit UI for Bodrum Plan Notları RAG Sistemi.

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import streamlit as st

from src.retriever import ask
from src.vector_db import VectorDB
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bodrum Plan Notları RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR BLAZING PREMIUM AESTHETICS ---
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1726 0%, #17202a 100%);
    }

    /* Titles */
    h1 {
        background: -webkit-linear-gradient(45deg, #0ed2f7, #b2ff05);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* User Message distinct style */
    .stChatMessage[data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background: linear-gradient(135deg, rgba(14, 210, 247, 0.1) 0%, rgba(178, 255, 5, 0.05) 100%);
        border: 1px solid rgba(14, 210, 247, 0.2);
    }
    
    /* Expander / Sources */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        font-weight: 500;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border-radius: 0 0 8px 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        padding: 15px;
        font-size: 0.9em;
        color: #cbd5e1;
    }
    
    .source-box {
        background: rgba(30, 41, 59, 0.7);
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        border-left: 3px solid #0ed2f7;
    }

    /* Input box floating */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 1px solid rgba(14, 210, 247, 0.3) !important;
        box-shadow: 0 0 20px rgba(14, 210, 247, 0.15) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {  /* Sidebar background */
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metrics / Status */
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- STATE AND INITIALISATION ---
@st.cache_resource(show_spinner=False)
def load_database() -> VectorDB:
    """Load the FAISS index once per session."""
    db = VectorDB()
    try:
        if db.exists():
            db.load()
            return db
    except Exception as e:
        logger.error(f"Error loading VectorDB: {e}")
    return None

db = load_database()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bodrum imar planları ve yapılaşma koşulları hakkında sorularınızı yanıtlamak için hazırım! Nasıl yardımcı olabilirim?"}
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Bodrum_Castle.jpg/800px-Bodrum_Castle.jpg", width="stretch")
    st.markdown("### 🏛️ Bodrum RAG Sistemi")
    st.caption("Bodrum Mimarlar Odası'nın plan notları üzerinde doğal dilde arama yapın.")
    
    st.divider()
    
    st.markdown("#### ⚙️ Sistem Durumu")
    if db is not None:
        st.success("✅ İndeks Hazır")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vektör Sayısı", f"{db.size}")
        with col2:
            st.metric("Boyut", f"{db.dimension}")
    else:
        st.error("❌ İndeks Bulunamadı")
        st.caption("Veritabanı oluşturmak için yandaki Yenile butonunu kullanın.")
    
    st.markdown("#### 🔧 Arama Ayarları")
    top_k = st.slider("Getirilecek Belge Sayısı (Top-K)", min_value=1, max_value=10, value=settings.top_k)
    
    st.divider()
    
    st.markdown("#### 🏗️ Veri Yönetimi")
    if st.button("🔄 Veritabanını Yenile (Re-Index)", help="Tüm belgeleri yeniden indirir, ayıklar ve indeksler. Uzun sürebilir."):
        with st.status("🚀 Pipeline çalıştırılıyor...", expanded=True) as status:
            try:
                from src.pipeline import run_pipeline
                st.write("Scraping ve PDF işlemleri başlatıldı...")
                # We skip re-downloading if files exist for speed during UI tests, 
                # but for a fresh deploy we'd want a full run.
                run_pipeline(skip_download=False)
                st.success("İndeksleme başarıyla tamamlandı!")
                status.update(label="✅ İndeksleme Tamamlandı!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Hata: {str(e)}")
                status.update(label="❌ Hata Oluştu", state="error")

    st.divider()
    st.caption("Mühendislik Harikası 🚀 Antigravity")


# --- MAIN CHAT UI ---
st.markdown("<h1>Bodrum Plan Notları RAG Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Mimarlar Odası plan belgelerinden yapay zeka destekli soru-cevap asistanı</div>", unsafe_allow_html=True)

if db is None:
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message and it has sources, display them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("📚 Kaynakları Görüntüle"):
                for idx, src in enumerate(message["sources"], 1):
                    score = src.get("score", 0)
                    url = src.get("source_url", "")
                    # Extract nice name from URL usually like "http...#pdf:NAME"
                    doc_name = url.split("#pdf:")[-1].replace("_", " ").replace(".pdf", "") if "#pdf:" in url else "Web Sayfası"
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📌 Kaynak {idx}: {doc_name}</strong>  
                        <span style="color:#64748b; font-size: 0.8em;">Benzerlik Skoru: {score:.3f}</span><br/>
                        <em>"{src.get('text_preview', '')}..."</em>
                    </div>
                    """, unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("Gümüşlük'te imar durumu nedir? Yokuşbaşı plan notlarında neler geçiyor?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("💭 Belgeler taranıyor ve analiz ediliyor..."):
            try:
                # Query the RAG system
                result = ask(db, prompt, top_k=top_k)
                
                answer = result.get("answer", "Beklenmedik bir hata oluştu.")
                sources = result.get("sources", [])
                
                # Show answer
                message_placeholder.markdown(answer)
                
                # Show sources immediately inside chat
                if sources:
                    with st.expander("📚 Kaynakları Görüntüle"):
                        for idx, src in enumerate(sources, 1):
                            score = src.get("score", 0)
                            url = src.get("source_url", "")
                            doc_name = url.split("#pdf:")[-1].replace("_", " ").replace(".pdf", "") if "#pdf:" in url else "Web Sayfası"
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>📌 Kaynak {idx}: {doc_name}</strong>  
                                <span style="color:#64748b; font-size: 0.8em;">Benzerlik Skoru: {score:.3f}</span><br/>
                                <em>"{src.get('text_preview', '')}..."</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                st.error(f"Bir hata oluştu: {str(e)}")
