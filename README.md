---

# Retrieval-Augmented Generation (RAG) Project

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system by loading PDF documents, processing them into manageable text chunks, storing embeddings in a vector database, and using a Large Language Model (LLM) to generate responses based on retrieved information.

## Project Overview

The project is divided into two main steps:
1. **Retrieval Step**: Loading and splitting documents, creating embeddings, and saving them in a vector database.
2. **Generation Step**: Using the **Google Generative Gemini Pro model** for generating responses based on document retrieval.

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.8+
- LangChain
- PyPDF2
- Sentence Transformers (for BERT model)
- Chroma
- Google Generative AI API (Gemini Pro model)

You can install the required packages using:

```bash
pip install langchain chromadb pypdf2 sentence-transformers
```

---

## Steps to Run the Project

### 1. **Retrieval Step**

#### 1.1 Load PDF Files
We use **LangChain's `DirectoryLoader`** and the **PyPDF loader** to load all PDF files in a specified folder:

```python
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# Load PDFs from a directory
pdf_loader = DirectoryLoader('<your-folder-path>', loader_cls=PyPDFLoader)
documents = pdf_loader.load()
```

#### 1.2 Text Splitting
To efficiently split the text into meaningful chunks, we use a **semantic text splitter** that employs a **BERT model** for better context retention:

```python
from langchain.text_splitter import SemanticTextSplitter
from transformers import BertTokenizer, BertModel

# Initialize BERT-based semantic text splitter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

splitter = SemanticTextSplitter(model, tokenizer, chunk_size=1000)
chunks = splitter.split_documents(documents)
```

#### 1.3 Create and Save Vector Database
The document chunks are embedded and stored in a **Chroma vector database**:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAiEmbeddings

# Initialize embeddings and vectorstore
embeddings = GoogleGenerativeAiEmbeddings(model="text-embedding-gpt")
vector_store = Chroma.from_documents(chunks, embedding=embeddings)

# Save the vector database
vector_store.save_local("<path-to-save-vector-store>")
```

### 2. **Generation Step**

#### 2.1 Load Vector Database and Perform Retrieval
Load the stored vector database and retrieve relevant documents using a similarity search:

```python
# Load vector store from saved location
vector_store = Chroma.load_local("<path-to-vector-store>", embedding=embeddings)

# Perform similarity search
query = "Your query here"
retrieved_docs = vector_store.similarity_search(query)
```

#### 2.2 Generate Response using Google Generative LLM
Pass the retrieved documents to the **Google Gemini Pro model** to generate a response:

```python
from langchain.llms import GoogleGenerativeAI

# Initialize Google Generative AI LLM
llm = GoogleGenerativeAI(model="gemini-pro")

# Generate response
response = llm.generate(query=query, context=retrieved_docs)
print(response)
```

---

## Project Workflow

1. **Load PDFs**: Load a folder containing PDF files using the `DirectoryLoader` and `PyPDFLoader`.
2. **Split Text**: Split the text into chunks using a semantic text splitter that uses a BERT model for meaningful segmentation.
3. **Create Vector Database**: Create embeddings for each chunk using Google Generative AI embeddings and store them in a Chroma vector database.
4. **Retrieve Documents**: Perform a similarity search to retrieve the most relevant documents for a query.
5. **Generate Response**: Use Google’s Gemini Pro model to generate a response based on the retrieved documents.

---

## Conclusion

This project provides a complete flow for building a Retrieval-Augmented Generation (RAG) system using LangChain, BERT for text splitting, Chroma for vector storage, and Google Generative AI for embeddings and generation. You can easily extend or modify this workflow based on your specific document retrieval and response generation needs.

---

Feel free to update this `README.md` with additional details as the project evolves!
