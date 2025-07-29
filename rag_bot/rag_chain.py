from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def rag_chain(query):
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", k=3)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False) #Change to true later

    return chain({"query": query})