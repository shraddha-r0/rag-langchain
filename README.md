# Ornithology RAG System

## Overview
Welcome to my experimental RAG (Retrieval-Augmented Generation) project! This repository serves as my learning playground where I'm building a system to work with my ornithology course notes using LangChain and ChromaDB.

## Project Goals
- Learn and experiment with RAG (Retrieval-Augmented Generation) techniques
- Build a system that can effectively retrieve and generate information from my ornithology course notes
- Implement vector search capabilities using ChromaDB
- Create a practical application of LLMs in the field of ornithology

## Tech Stack
- **LangChain**: For building applications with LLMs
- **ChromaDB**: As the vector database for efficient similarity search
- **Python**: The primary programming language
- **Jupyter Notebooks**: For experimentation and visualization

## Project Structure
- `rag/`: Virtual environment directory (not tracked by git)
- `pdfs/`: Directory containing ornithology course notes (not tracked by git)
- `chroma_db/`: Vector database storage (not tracked by git)
- `rag_implementation.ipynb`: Main notebook for the RAG implementation
- `config.py`: Configuration settings
- `requirements.txt`: Python dependencies

## Getting Started
1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv rag
   source rag/bin/activate  # On Windows: rag\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your course notes to the `pdfs/` directory
5. Start exploring with the Jupyter notebook:
   ```bash
   jupyter notebook rag_implementation.ipynb
   ```

## Note
This is an experimental project and work in progress. The code and structure may change frequently as I continue to learn and improve the system.

## License
This project is for educational purposes only.
