# RAG Pipeline

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

## Chunking Strategies

Below are examples of different chunking strategies applied to the beginning of Alice in Wonderland:

### Original Text - 180 words 960 characters

```
  `Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling down stairs!  How brave they'll
all think me at home!  Why, I wouldn't say anything about it,
even if I fell off the top of the house!' (Which was very likely
true.)

  Down, down, down.  Would the fall NEVER come to an end!  `I
wonder how many miles I've fallen by this time?' she said aloud.
`I must be getting somewhere near the centre of the earth.  Let
me see:  that would be four thousand miles down, I think--' (for,
you see, Alice had learnt several things of this sort in her
lessons in the schoolroom, and though this was not a VERY good
opportunity for showing off her knowledge, as there was no one to
listen to her, still it was good practice to say it over) `--yes,
that's about the right distance--but then I wonder what Latitude
or Longitude I've got to?'  (Alice had no idea what Latitude was,
or Longitude either, but thought they were nice grand words to
say.)
```

### Fixed Size Chunking - 100 Characters

```
[10 Chunks Generated]

--- Chunk 1 ---
  `Well!' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling do

--- Chunk 2 ---
wn stairs!  How brave they'll all think me at home!  Why, I wouldn't say anything about it, even if 

--- Chunk 3 ---
I fell off the top of the house!' (Which was very likely
true.)

  Down, down, down.  Would the fall

--- Chunk 4 ---
 NEVER come to an end!  `I
wonder how many miles I've fallen by this time?' she said aloud. `I must 

--- Chunk 5 ---
be getting somewhere near the centre of the earth.  Let me see:  that would be four thousand miles d
	.
	.
	.
```

### Sentence Chunking
```
[10 Chunks Generated]

--- Chunk 1 ---
  `Well!' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling down stairs!  

--- Chunk 2 ---
How brave they'll all think me at home!  

--- Chunk 3 ---
Why, I wouldn't say anything about it, even if I fell off the top of the house!'

--- Chunk 4 ---
(Which was very likely true.)

--- Chunk 5 ---
Down, down, down.
	.
	.
	.
```

### Paragraph Chunking

```
[2 Chunks Generated]

--- Chunk 1 ---
Alice was beginning to get very tired of sitting by her sister
on the bank, and of having nothing to do:  once or twice she had
peeped into the book her sister was reading, but it had no
pictures or conversations in it, `and what is the use of a book,'
thought Alice `without pictures or conversation?'

--- Chunk 2 ---
So she was considering in her own mind (as well as she could,
for the hot day made her feel very sleepy and stupid), whether
the pleasure of making a daisy-chain would be worth the trouble
of getting up and picking the daisies, when suddenly a White
Rabbit with pink eyes ran close by her.
```

### Token-Limited Chunking 512 Tokens + 10 Token Overlap

```
[3 Chunks Generated]

--- Chunk 1 ---
`Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling down stairs!  How brave they'll
all think me at home!  Why, I wouldn't say anything about it,
even if I fell off the top of the house!' (Which was very likely
true.)

  Down, down, down.  Would the fall NEVER come to an end!  `I
wonder how many miles I've fallen by

--- Chunk 2 ---
wonder how many miles I've fallen by this time?' she said aloud.
`I must be getting somewhere near the centre of the earth.  Let
me see:  that would be four thousand miles down, I think--' (for,
you see, Alice had learnt several things of this sort in her
lessons in the schoolroom, and though this was not a VERY good
opportunity for showing off her knowledge, as there was no one to
listen to her,

--- Chunk 3 ---
there was no one to
listen to her, still it was good practice to say it over) `--yes,
that's about the right distance--but then I wonder what Latitude
or Longitude I've got to?'  (Alice had no idea what Latitude was,
or Longitude either, but thought they were nice grand words to
say.)
```


## Author
**Joe12827**

