# rag_bot/build_vector_db.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

PDF_DIR = "pdfs"
CHROMA_DIR = "chroma_db"

def load_pdfs(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Loaded {filename} with {len(docs)} pages")
    return all_docs

def split_docs(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"🔪 Split into {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks, persist_dir):
    # Delete existing database if it exists
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Using a smaller model
    vectordb = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"💾 Vector DB persisted at '{persist_dir}'")

def build_vector_db():
    print("🚀 Starting vector DB build...")
    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs)
    embed_and_store(chunks, CHROMA_DIR)
    print("🎉 Vector DB build complete!")

if __name__ == "__main__":
    build_vector_db()

    