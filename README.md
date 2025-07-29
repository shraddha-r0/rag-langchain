# 🦜 Ornithology RAG Chatbot

## 📚 Overview

Welcome to the Ornithology RAG (Retrieval-Augmented Generation) Chatbot! This project is a sophisticated question-answering system that leverages the power of large language models (LLMs) and vector databases to provide accurate and engaging responses to questions about ornithology. The system uses a combination of document retrieval and language generation to deliver well-informed answers based on your course materials.

## ✨ Features

- **Dual-Tone Responses**: Choose between factual or witty response styles
- **Source-Cited Answers**: View the exact sources used to generate each response
- **Efficient Retrieval**: Uses ChromaDB for fast semantic search
- **Query Optimization**: Automatically refines questions for better search results
- **Interactive Web Interface**: User-friendly Streamlit-based UI

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for efficient document retrieval
- **OpenAI**: For embeddings (text-embedding-3-small) and chat completions (GPT-3.5-turbo)
- **Streamlit**: For the web interface
- **Python 3.8+**: Core programming language

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit web application
├── build_vector_db.py      # Script to create the vector database
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── pdfs/                   # Directory for PDF course notes (not tracked by git)
│   └── your_notes.pdf     # Example: Add your PDFs here
├── chroma_db/             # Vector database storage (not tracked by git)
└── rag_bot/
    └── rag_chain.py       # Core RAG implementation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- PDF documents containing ornithology content

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ornithology-rag.git
   cd ornithology-rag
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   - Copy `.env.example` to `.env`
   - Add your OpenAI API key:

     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

5. **Add your PDFs**

   - Place your ornithology PDFs in the `pdfs/` directory
   - The system will automatically process all PDFs in this directory

## 🏗️ Building the Vector Database

Before running the application, you need to build the vector database:

```bash
python build_vector_db.py
```

This script will:

1. Load all PDFs from the `pdfs/` directory
2. Split them into manageable chunks
3. Generate embeddings using OpenAI's text-embedding-3-small model
4. Store the vectors in the `chroma_db/` directory

## 🖥️ Running the Application

### Local Development

Start the Streamlit web interface locally with:

```bash
streamlit run app.py
```

Then open your browser to the provided URL (typically `http://localhost:8501`).

### Live Demo

A live version of the application is available at:
🔗 [https://rag-bird-bot.streamlit.app](https://rag-bird-bot.streamlit.app)

## 🧠 How It Works

1. **Document Processing**

   - PDFs are loaded and split into chunks
   - Each chunk is converted to a vector embedding
   - Vectors are stored in ChromaDB for efficient search

2. **Query Processing**

   - User questions are transformed for better retrieval
   - The system finds the most relevant document chunks
   - The LLM generates a response using the retrieved context

3. **Response Generation**

   - The system can generate responses in different tones
   - All answers are grounded in the provided documents
   - Sources are cited for verification

## 📝 File Descriptions

### `app.py`

The Streamlit web interface that provides a user-friendly way to interact with the RAG system. Handles user input, displays responses, and shows source documents.

### `build_vector_db.py`

Script that processes PDF documents, splits them into chunks, generates embeddings, and stores them in a ChromaDB vector database.

### `rag_bot/rag_chain.py`

Core implementation of the RAG system. Handles query transformation, document retrieval, and response generation using LangChain and OpenAI.

## 🔧 Customization

### Changing the Response Tone

Edit the system prompts in `rag_bot/rag_chain.py` to modify the response styles or add new ones.

### Adjusting Chunking

Modify the `split_docs` function in `build_vector_db.py` to change how documents are split:

- `chunk_size`: Maximum size of each text chunk (default: 500 characters)
- `chunk_overlap`: Overlap between chunks (default: 100 characters)

## 📊 Performance Considerations

- **Embedding Model**: Using `text-embedding-3-small` for a good balance of performance and cost
- **Retrieval**: The system retrieves the top 3 most relevant document chunks by default
- **Caching**: The vector database is persisted to disk for faster subsequent loads

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Note that while the code is open source, you'll need your own OpenAI API key to run the application.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- LLM services provided by [OpenAI](https://openai.com/)
- Web interface built with [Streamlit](https://streamlit.io/)
