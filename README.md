
# Retrieval-Augmented Generation (RAG) System with LangChain

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain** and **LangChain Expression Language (LCEL)**.  
The system answers user questions by retrieving relevant information from a curated set of personal documents and then generating grounded responses using a Large Language Model (LLM).

Unlike a standard chatbot that relies only on pretrained knowledge, this RAG system ensures **fact-based, document-grounded answers** with **explicit source citations** and **conversational memory** for follow-up questions.

---

## Problem Statement

Large Language Models can generate fluent answers, but they:
- May hallucinate facts
- Cannot access private or custom documents
- Lose context across multiple questions

### Problem Solved

This project addresses these challenges by:
- Retrieving relevant document chunks using embeddings
- Grounding responses strictly in retrieved documents
- Maintaining conversational context across turns
- Providing source citations for transparency and trust

---

## Document Collection

### Documents Used

The system uses **five personal documents** representing a single AI Engineer persona:

1. Personal Biography  
2. Professional Resume  
3. AI & Machine Learning Projects  
4. Teaching & Mentorship Experience  
5. Research Interests & Career Goals  

### Why These Documents?

- They are semantically related but not redundant  
- They cover skills, experience, projects, and goals  
- They simulate a realistic private knowledge base  
- They are ideal for evaluating retrieval and chunking strategies  

---

## System Architecture

### High-Level Workflow

1. User submits a question  
2. The question is embedded  
3. Relevant document chunks are retrieved from a vector store  
4. Retrieved context and question are injected into a custom prompt  
5. LLM generates a grounded answer  
6. Sources are cited  
7. Conversation history is stored for follow-up questions  

---

## Technologies Used

- Python  
- LangChain  
- LangChain Expression Language (LCEL)  
- ChromaDB Vector Store  
- OpenAI Embeddings  
- ChatOpenAI  
- Conversational Memory  

---

## How to Run the System

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd rag-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 4. Run the Notebook
```bash
jupyter notebook rag_system.ipynb
```

---

## Example Interactions

### Example 1: Direct Question

**User:**  
What AI projects has Olasunkanmi worked on?

**System:**  
Olasunkanmi Akeem Rasak has worked on the following AI projects:
- Sales forecasting
- Car price prediction
- Student performance analytics
- Forex trading strategy automation
- Hybrid car scanner automation
- Document-based AI systems

**Sources:**  
- documents\Personal Biography.pdf
- documents\Professional Resume.pdf

---

### Example 2: Follow-Up Question (Conversational Memory)

**User:**  
Which of those involve RAG systems?

**System:**  
Olasunkanmi Akeem Rasak has worked on the following projects involving Retrieval-Augmented Generation (RAG) systems:
- Retrieval-Augmented Generation (RAG) Systems
- Retrieval-Augmented Generation (RAG) Chatbot

**Sources:** 
- documents\Professional Resume.pdf
- documents\AI & ML Projects.pdf

---

## What I Learned

Through this project, I learned:
- How RAG systems reduce hallucinations by grounding responses in documents
- The impact of chunk size and overlap on retrieval quality
- How to build LCEL-based pipelines
- How to design effective prompts for document-grounded answers
- How conversational memory improves user interaction
- How to structure a production-style RAG workflow

---

## Future Improvements

Potential enhancements include:
- Hybrid retrieval (BM25 + vector search)
- Sentence-level source attribution
- Web or Streamlit-based user interface
- Retrieval evaluation metrics
- Multi-user session memory
- Reranking for improved document relevance

---

## Conclusion

This project demonstrates a complete end-to-end **Retrieval-Augmented Generation system** built using modern LangChain practices.  
It highlights how LLMs can be combined with private knowledge bases to create accurate, explainable, and conversational AI systems suitable for real-world applications.
