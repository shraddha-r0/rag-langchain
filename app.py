import streamlit as st
import os
from dotenv import load_dotenv
from rag_bot.rag_chain import rag_chain
load_dotenv()

st.set_page_config(page_title="🦜 Ornithology RAG Bot")

st.title("🦜 Ornithology RAG Chatbot")
st.markdown("Ask me anything about birds!")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking like a clever parrot..."):
        result = rag_chain(query)
        st.markdown("**Answer:**")
        st.write(result["result"])