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
	- `Qwen3-Embedding-4B`
- **Vector Database:**
	- `Pinecone` (for storing and searching embeddings)
- **LLM Integration:**
	- `Llama-3` (From HuggingFace)
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

### Fixed Size Chunking: 100 Characters

```
[10 Chunks Generated]

--- Chunk 1 ---
  `Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling do

--- Chunk 2 ---
wn stairs!  How brave they'll
all think me at home!  Why, I wouldn't say anything about it,
even if 

--- Chunk 3 ---
I fell off the top of the house!' (Which was very likely
true.)

  Down, down, down.  Would the fall

--- Chunk n ---
	.
	.
	.
```

### Sentence Chunking
```
[10 Chunks Generated]

--- Chunk 1 ---
  `Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling down stairs!  

--- Chunk 2 ---
How brave they'll
all think me at home!  

--- Chunk 3 ---
Why, I wouldn't say anything about it,
even if I fell off the top of the house!'

--- Chunk n ---
	.
	.
	.
```

### Paragraph Chunking

```
[2 Chunks Generated]

--- Chunk 1 ---
`Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling down stairs!  How brave they'll
all think me at home!  Why, I wouldn't say anything about it,
even if I fell off the top of the house!' (Which was very likely
true.)

--- Chunk 2 ---
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

### [Token-Limited Chunking](#https://github.com/Joe12827/RAG-Pipeline/blob/e48a47fe9318639df21d6a99b3f32d521ff8715b/src/chunker.py#L38): 100 Tokens + 10 Token Overlap

```
[3 Chunks Generated]

--- Chunk 1 ---
`Well!' thought Alice to herself, `after such a fall as this, I
shall think nothing of tumbling down stairs!  How brave they'll
all think me at home!  Why, I wouldn't say anything about it,
even if I fell off the top of the house!' (Which was very likely
true.) Down, down, down.  Would the fall NEVER come to an end!  `I
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

## Embedding
Once the documents are seperated into chunks, embeddings are created
```python
Document(
	text="`Well!' thought Alice to herself, `after such a fa...",
	vector=array([ 0.00016985, -0.02154499, ..., 0.00395041,  0.00889554],
		shape=(2560,),
		dtype=float32),
	source='alice_in_wonderland_short.txt')
```

## Retrieval

When a question is asked, the query is tokenized and the top-k results are retreived from the vector database (Pinecone)
Then, the query and documents gathered are fed into an LLM.
```
Query: What happens to Alice after she drinks from the bottle labeled "Drink Me"?

chunk-12: Score: 0.5194
chunk-5:  Score: 0.5010
chunk-18: Score: 0.4938
chunk-44: Score: 0.4789
chunk-58: Score: 0.4637

Feeding context to LLM...

Answer: After Alice drinks from the bottle labeled "Drink Me," she finds that it has a very
mixed flavor of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast.
She finishes it off quickly, and then notices that she has shrunk down to a tiny size, only 10
inches high.
```

## Author
**Joe Wesnofske - Joe12827**

