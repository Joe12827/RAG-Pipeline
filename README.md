# RAG Pipeline Portfolio

## Overview
The RAG Pipeline is a modular and extensible Retrieval-Augmented Generation (RAG) system designed to process, chunk, and embed documents from various formats. It enables efficient document ingestion and semantic search, making it ideal for building advanced AI-powered retrieval and question-answering systems.

## Features
- **Multi-format Document Ingestion:** Supports TXT, HTML, PDF, and DOCX files.
- **Automatic Preprocessing:** Selects the correct preprocessor for each file type.
- **Flexible Chunking Strategies:** Easily swap chunking strategies for different use cases.
- **Embedding Generation:** Converts document chunks into semantic embeddings for downstream tasks.
- **Traceable Results:** Returns embeddings with their source filenames for easy tracking.
- **Vector Database Integration:** Store and search embeddings using Pinecone.
- **LLM Integration:** Use HuggingFace models for advanced language tasks.

## Technologies & Libraries Used
- **Python 3.10+**
- **Document Processing:**
	- `python-docx` (DOCX extraction)
	- `PyPDF2` or `pdfminer.six` (PDF extraction)
	- `BeautifulSoup4` (HTML extraction)
- **Chunking:** Custom strategies (sentence, paragraph, etc.)
- **Embedding:**
	- `Qwen3` (for embedding generation)
- **Vector Database:**
	- `pinecone-client` (for storing and searching embeddings)
- **LLM Integration:**
	- `transformers` (HuggingFace for LLMs)
- **Other:**
	- `os` (file handling)
	- `logging` (optional, for debug output)

## Example Usage
```python
from ingestion_pipeline import IngestionPipeline
from chunker import TokenLimitedChunking
from embedder import Qwen3Embedder

embedder = Qwen3Embedder()
chunker = TokenLimitedChunking(embedder.tokenizer, max_tokens=512, overlap_tokens=10)

pipeline = IngestionPipeline(chunker=chunker, embedder=embedder)

files = ["Data/sample.txt", "Data/sample.pdf"]
pipeline.process_documents(files)
```

## Author
**Joe12827**

