"""
RAG (Retrieval-Augmented Generation) Chain Implementation

This module implements the core RAG functionality for the Ornithology Chatbot.
It handles query transformation, document retrieval, and response generation
using LangChain and OpenAI's language models.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Verify that the OpenAI API key is set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please set it in your .env file as OPENAI_API_KEY=your_api_key_here"
    )

# Initialize the language model for generating responses
# Using gpt-3.5-turbo for a good balance of performance and cost
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Model to use for response generation
    temperature=0,  # Lower temperature for more focused and deterministic responses
    api_key=OPENAI_API_KEY  # Pass the API key explicitly for clarity
)

# Initialize the embeddings model for document and query encoding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Efficient model for embeddings
    api_key=OPENAI_API_KEY
)

# Initialize the Chroma vector store for document retrieval
# The vector store must be pre-populated using build_vector_db.py
vectordb = Chroma(
    persist_directory="chroma_db",  # Directory where the vector store is saved
    embedding_function=embeddings   # Function to generate embeddings for queries
)

# Define a prompt template for query transformation
# This helps convert natural language questions into effective search queries
query_transform_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""
    You are an expert assistant that transforms user questions into clear, 
    unambiguous search queries to help retrieve the most relevant information 
    from a knowledge base about ornithology.

    Your job is to:
    - Clarify vague or incomplete questions.
    - Remove irrelevant words or noise.
    - Preserve the user's intent and specificity.
    - Output a standalone, well-formed query optimized for semantic retrieval.

    Follow these guidelines:
    - Be concise and specific.
    - Remove filler phrases (e.g., "can you tell me", "I want to know about...").
    - Expand acronyms or shorthand when appropriate.
    - Include relevant context if mentioned.
    - Do not answer the question—only rewrite it.

    Original query: {original_query}
    Transformed query:"""
)

# Create a chain that first transforms the query using the prompt template,
# then sends it to the language model for processing
query_transformer = query_transform_prompt | llm

def build_rag_chain(tone: str = "factual") -> RetrievalQA:
    """
    Constructs a RetrievalQA chain with a prompt template based on the specified tone.
    
    Args:
        tone (str): The desired tone for responses. Can be "factual" or "witty".
                   Defaults to "factual".
                   
    Returns:
        RetrievalQA: A configured retrieval-augmented generation chain.
    """
    # Define the system prompt based on the selected tone
    if tone == "witty":  # Humorous and engaging responses
        system_prompt = """
        You are a knowledgeable but witty study buddy who loves birds. 
        Your answers are clear, fact-based, and have a light touch of humor or clever commentary.
        Do not make things up—if the answer isn't in the context, say "I don't know."
        Keep responses concise but engaging.
        """
    else:  # Default to factual and concise responses
        system_prompt = """
        You are a concise and accurate ornithology assistant. 
        Answer questions based strictly on the given context. 
        Do not add humor or filler. If you don't know, say "I don't know."
        """

    # Create a prompt template that combines the system prompt with the context and question
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        {system_prompt}

        Context: {{context}}

        Question: {{question}}

        Answer:
        """
    )

    # Build and return the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,  # Language model for generating responses
        retriever=vectordb.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        ),
        return_source_documents=True,  # Include source documents in the response
        chain_type="stuff",  # Simple chain type that stuffs all documents into the prompt
        chain_type_kwargs={
            "prompt": prompt_template,  # Use our custom prompt template
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"  # Simple template for formatting document content
            )
        }
    )

def get_rag_response(query: str, tone: str = "factual") -> Dict[str, Any]:
    """
    Process a user query through the RAG pipeline and return the response.
    
    This function:
    1. Transforms the user's query for better retrieval
    2. Builds a RAG chain with the specified tone
    3. Executes the chain with the transformed query
    4. Returns the result with answer and source documents
    
    Args:
        query (str): The user's question about ornithology
        tone (str): The desired response tone ("factual" or "witty"). 
                   Defaults to "factual".
                   
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'result': The generated answer
            - 'source_documents': List of source documents used for the answer
    """
    # Transform the user's query to improve retrieval
    transformed_query = query_transformer.invoke({
        "original_query": query
    }).content.strip('"\' \n')
    
    # Log the original and transformed queries for debugging
    print(f"🧾 Original query: {query}")
    print(f"🔍 Transformed query: {transformed_query}")

    # Build the RAG chain with the specified tone
    chain = build_rag_chain(tone)
    
    # Execute the chain with both the transformed query (for retrieval)
    # and the original question (for the final response)
    result = chain.invoke({
        "query": transformed_query,  # Used for document retrieval
        "question": query            # Used for generating the final response
    })
    
    return result