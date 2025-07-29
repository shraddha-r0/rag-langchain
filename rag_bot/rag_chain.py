import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ✅ Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")

# ✅ LLM + Embeddings
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# ✅ Vector store
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# ✅ Query transformation prompt (moved to top)
query_transform_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""
    You are an expert assistant that transforms user questions into clear, unambiguous search queries to help retrieve the most relevant information from a knowledge base.

    Your job is to:
    - Clarify vague or incomplete questions.
    - Remove irrelevant words or noise.
    - Preserve the user’s intent and specificity.
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

# ✅ Chain to transform query
query_transformer = query_transform_prompt | llm

# ✅ Function to dynamically build RAG chain with chosen tone
def build_rag_chain(tone: str):
    if tone == "witty":
        system_prompt = """
        You are a knowledgeable but witty study buddy who loves birds. 
        Your answers are clear, fact-based, and have a light touch of humor or clever commentary.
        Do not make things up—if the answer isn't in the context, say “I don’t know.”
        Keep responses concise but engaging.
        """
    else:
        system_prompt = """
        You are a concise and accurate ornithology assistant. 
        Answer questions based strictly on the given context. 
        Do not add humor or filler. If you don't know, say “I don’t know.”
        """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        {system_prompt}

        Context: {{context}}

        Question: {{question}}

        Answer:
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt_template,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        }
    )

# ✅ Final wrapper function
def get_rag_response(query: str, tone: str = "factual") -> dict:
    transformed_query = query_transformer.invoke({"original_query": query}).content.strip('"\' \n')
    print(f"🧾 Original query: {query}")
    print(f"🔍 Transformed query: {transformed_query}")

    chain = build_rag_chain(tone)
    result = chain.invoke({"query": transformed_query, "question": query})
    return result