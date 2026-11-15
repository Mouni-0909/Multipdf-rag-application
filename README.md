# Multi PDF RAG Application

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application built using Python, LangChain, FAISS, HuggingFace Embeddings, and OpenAI.

It allows users to ask questions from PDF documents and get answers based on the content available in those documents.

The project contains two applications:

1. **Single PDF RAG (`rag_app.py`)**
   - Loads one PDF document.
   - Extracts and splits the text into smaller chunks.
   - Creates embeddings and stores them in FAISS.
   - Retrieves relevant information to answer user questions.

2. **Multi PDF RAG (`multi_pdf_rag_app.py`)**
   - Loads multiple PDF documents from the `pdfs` folder.
   - Combines the content of all PDFs.
   - Retrieves relevant information from different documents.
   - Displays the PDF filename and page number used for the answer.

## Technologies Used

- Python
- LangChain
- OpenAI (GPT-4o-mini)
- HuggingFace Embeddings
- FAISS Vector Database
- PyPDFLoader
- Python dotenv
- LangSmith for tracing

## Project Structure

```text
Multi_PDF_RAG_app/
│
├── pdfs/
│   ├── internetsetup.pdf
│   ├── javascript_tutorial.pdf
│   └── PYTHON PROGRAMMING NOTES.pdf
│
├── rag_app.py
├── multi_pdf_rag_app.py
├── README.md
├── .gitignore
└── .env
```

## Setup Instructions

### 1. Create and activate virtual environment

```bash
python -m venv newvenv
newvenv\Scripts\activate
```

### 2. Install required libraries

```bash
pip install langchain langchain-community langchain-text-splitters langchain-openai langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv
```

### 3. Configure API keys

Create a `.env` file in the project folder and add:

```env
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Multi_PDF_RAG_Project
```

### 4. Run the Single PDF application

```bash
py rag_app.py
```

### 5. Run the Multi PDF application

```bash
py multi_pdf_rag_app.py
```

## How It Works

1. Load PDF documents.
2. Extract text from PDFs.
3. Split text into smaller chunks.
4. Generate embeddings using HuggingFace.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on the user's question.
7. Send the retrieved context to OpenAI.
8. Generate an answer with relevant PDF source information.

## Key Features

- Supports single and multiple PDF documents.
- Semantic search using vector embeddings.
- Context-based answers using OpenAI.
- Displays source PDF names and page numbers.
- Local FAISS vector database storage.
- LangSmith integration for tracing and monitoring.

## Learning Outcome

Through this project, I learned how to build a basic RAG application, work with PDF documents, generate embeddings, use FAISS for similarity search, integrate OpenAI, and implement multi-document question answering.


