"""
Vector Database Builder for Ornithology RAG System

This script processes PDF documents from a specified directory, splits them into chunks,
and creates a vector database using ChromaDB with OpenAI embeddings.
The resulting vector database is used for efficient similarity search in the RAG system.
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Constants
PDF_DIR = "pdfs"  # Directory containing PDF files to process
CHROMA_DIR = "chroma_db"  # Directory to store the vector database

def load_pdfs(pdf_dir: str) -> list:
    """
    Load and extract text from all PDF files in the specified directory.
    
    Args:
        pdf_dir (str): Path to the directory containing PDF files
        
    Returns:
        list: List of document objects containing text content and metadata
    """
    all_docs = []
    # Process each PDF file in the directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()  # Load all pages from the PDF
            all_docs.extend(docs)
            print(f"✅ Loaded {filename} with {len(docs)} pages")
    return all_docs

def split_docs(docs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        docs (list): List of document objects to split
        chunk_size (int): Maximum size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    # Initialize the text splitter with specified chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪 Split into {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks: list, persist_dir: str) -> None:
    """
    Create vector embeddings for document chunks and store them in ChromaDB.
    
    Args:
        chunks (list): List of document chunks to embed
        persist_dir (str): Directory to store the vector database
    """
    # Remove existing vector database if it exists
    if os.path.exists(persist_dir):
        print(f"♻️  Removing existing vector database at '{persist_dir}'")
        shutil.rmtree(persist_dir)
    
    # Initialize the embedding model
    # Using text-embedding-3-small for better performance and lower cost
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create and persist the vector database
    print(f"🧠 Creating embeddings and building vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"💾 Vector database persisted at '{persist_dir}'")

def build_vector_db() -> None:
    """
    Main function to build the vector database from PDF documents.
    Orchestrates the loading, splitting, and embedding of documents.
    """
    print("🚀 Starting vector database build process...")
    
    # Step 1: Load documents from PDF files
    print("📄 Loading PDF documents...")
    docs = load_pdfs(PDF_DIR)
    if not docs:
        print("❌ No PDF files found in the 'pdfs' directory")
        return
    
    # Step 2: Split documents into chunks
    print("✂️  Splitting documents into chunks...")
    chunks = split_docs(docs)
    
    # Step 3: Create embeddings and store in vector database
    print("🔧 Processing chunks and creating embeddings...")
    embed_and_store(chunks, CHROMA_DIR)
    
    print("✅ Vector database build complete!")
    print(f"✨ Total documents processed: {len(docs)}")
    print(f"✨ Total chunks created: {len(chunks)}")

if __name__ == "__main__":
    # Execute the vector database build process when run as a script
    build_vector_db()