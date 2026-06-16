# AI Document Assistant (RAG)

## Overview

AI Document Assistant is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions in natural language.

The system extracts text from PDF files, converts the content into embeddings using Sentence Transformers, stores them in a FAISS vector database, retrieves the most relevant document sections, and generates accurate answers using Gemini 2.5 Flash.

---

## Features

* PDF Upload and Processing
* Text Extraction from Documents
* Intelligent Text Chunking
* Semantic Search using Embeddings
* FAISS Vector Database
* Context-Aware Question Answering
* Gemini 2.5 Flash Integration
* Interactive Streamlit Interface

---

## Technologies Used

* Python
* Streamlit
* PyPDF
* Sentence Transformers
* FAISS
* LangChain Text Splitters
* Google Gemini 2.5 Flash

---

## How It Works

1. Upload a PDF document.
2. Extract text from the document.
3. Split text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in FAISS.
6. Ask a question related to the document.
7. Retrieve the most relevant chunks.
8. Send retrieved context to Gemini AI.
9. Generate and display the final answer.

---

## Project Architecture

PDF Document → Text Extraction → Text Chunking → Embeddings Generation → FAISS Vector Database → Similarity Search → Gemini 2.5 Flash → Answer Generation

---

## Example Questions

* What is the refund policy?
* What are the user responsibilities?
* How can an account be terminated?
* What payment conditions are mentioned?
* What services are covered in the agreement?

---

## Key Concepts

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* Similarity Search
* Large Language Models (LLMs)
* Document Question Answering

---

## Author

Eisha Younas

Artificial Intelligence Student | Machine Learning Enthusiast | Generative AI Developer

