#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder


# In[2]:


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print("API key loaded")


# #### Documents collections

# In[3]:


documents = []

for pdf_path in glob.glob("documents/*.pdf"):  # adjust folder path
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    documents.extend(docs)

print(f"Loaded {len(documents)} pages from PDFs")


# #### Text Splitters

# In[4]:


# Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len
)

# Split documents
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: {chunk.page_content}")


# #### Embeddings

# In[5]:


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# Test embedding
test_embedding = embeddings.embed_query("What is RAG?")
print(f"Embedding dimension: {len(test_embedding)}")
print(f"First 5 values: {test_embedding[:5]}")


# #### Vector Store
# # Create vector store from documents

# In[6]:


vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="my_info_collection",
    persist_directory="./chroma_db"
)


# In[7]:


# Test retriver
query = "Technical skills"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

results = retriever.invoke(query)
results


# #### Conversational Rag

# In[8]:


# Create LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# prompt
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions about Alex Morgan using the provided documents.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

<context>
{context}
</context>

Question: {question}

Answer in clear sentences.
At the end, list the sources you used as bullet points.
""")

# format documents
def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )

# RAG chain Using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# In[9]:


query = "What AI projects has Alex worked on?"
response = rag_chain.invoke(query)
print(response)


# #### Conversational RAG

# In[10]:


# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.output_parsers import StrOutputParser

# # Session store
# chat_store = {}

# def get_session_history(session_id: str):
#     if session_id not in chat_store:
#         chat_store[session_id] = InMemoryChatMessageHistory()
#     return chat_store[session_id]

# # Prompt
# conversational_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant answering questions about Alex Morgan."),
#     MessagesPlaceholder("chat_history"),
#     ("system", "Use the document context to answer.\nContext:\n{context}"),
#     ("user", "{question}")
# ])

# # Chain
# conversational_chain = (
#     {
#         "context": retriever | format_docs,
#         "question": lambda x: x["question"]
#     }
#     | conversational_prompt
#     | llm
#     | StrOutputParser()
# )

# # Memory wrapper
# rag_with_memory = RunnableWithMessageHistory(
#     conversational_chain,
#     get_session_history,
#     input_messages_key="question",
#     history_messages_key="chat_history"
# )


# In[ ]:





# In[11]:


# Store for chat histories
chat_store = {}

def get_session_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]

# Create conversational prompt
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant answering questions about Alex Morgan using the provided documents. Use ONLY the context below to answer the question. If the answer is not in the context, say I don't know."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Answer in clear sentences. At the end, list the sources you used as bullet points."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# Build base chain
conv_chain_base = (
    RunnableParallel(
        context=lambda x: format_docs(retriever.invoke(x["question"])),
        question=lambda x: x["question"],
        chat_history=lambda x: x.get("chat_history", [])
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)

# Wrap with message history
conv_chain = RunnableWithMessageHistory(
    conv_chain_base,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


# 
# **Questions**

# In[12]:


# First question
response = conv_chain.invoke(
    {"question": "What projects has Alex worked on?"},
    config={"configurable": {"session_id": "user_1"}}
)
print("Response 1:\n", response)

# Follow-up question
response2 = conv_chain.invoke(
    {"question": "Which of those involve RAG systems?"},
    config={"configurable": {"session_id": "user_1"}}
)

print("\nResponse 2:\n", response2)


# In[ ]:




