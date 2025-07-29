import streamlit as st
import os
from dotenv import load_dotenv
from rag_bot.rag_chain import get_rag_response
import logging

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/chatbot_usage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# Page config
st.set_page_config(page_title="🦜 Ornithology RAG Chatbot", page_icon="🦜", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📚 Study with the Bird Brain Bot</h1>
    <p style='text-align: center; font-size:18px;'>Learning ornithology? I’m your feathered friend with fun facts, witty quips, and no judgment for last-minute cramming.</p>
    """, unsafe_allow_html=True)
st.markdown("---")
# Add tone selector
tone = st.radio(
    "Choose the response tone:",
    ["Factual", "Witty"],
    horizontal=True
)
tone_key = tone.lower().replace(" ", "_")  # "factual" or "witty"
query = st.text_input("📝 What’s your bird-related question?", placeholder="e.g., How do hummingbirds hover?")

if query:
    with st.spinner("Consulting the aviary archives... 🪶"):
        response = get_rag_response(query, tone=tone_key)
        answer = response["result"]
        sources = response["source_documents"]

        # Logging
        logging.info(f"USER: {query}")
        logging.info(f"BOT: {answer}")

        # Output
        st.markdown("### 🧠 Answer")
        st.write(answer)

        # Sources in expandable
        with st.expander("🔍 View sources used in this answer"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content[:400] + "...")