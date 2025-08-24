import spacy
from abc import ABC, abstractmethod
from tqdm import tqdm
import re


# Strategy Interface
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass

    def normilize_text(text: str) -> str:
            # Replace line breaks that are not followed by another line break
            # (i.e., within a paragraph) with a space
            text = re.sub(r'\n(?!\n)', ' ', text)
            # Optionally, collapse multiple spaces
            text = re.sub(r'\s+', ' ', text)
            return text.strip()


# Concrete Strategy 1 — Fixed-size character chunks
class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        normilized_chunks = [ChunkingStrategy.normilize_text(chunk) for chunk in chunks]
        return normilized_chunks


# Concrete Strategy 2 — Sentence-based chunking
class SentenceChunking(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        normilized_sentences = [ChunkingStrategy.normilize_text(sentence) for sentence in sentences]
        return normilized_sentences


# Concrete Strategy 3 — Paragraph-based chunking
class ParagraphChunking(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        normilized_paragraphs = [ChunkingStrategy.normilize_text(paragraph) for paragraph in paragraphs]
        return normilized_paragraphs


# Concrete Strategy 4 — Token-limited chunking
class TokenLimitedChunking(ChunkingStrategy):
    """
    Note: This strategy returns tokenized chunks directly suitable for model input.
    It uses the tokenizer's `max_length` and `stride` parameters to create overlapping chunks.
    512 tokens is a good default for many models, but can be adjusted.
    """
    def __init__(self, tokenizer, max_tokens=512, overlap_tokens=10):
        if overlap_tokens >= max_tokens:
            raise ValueError("Overlap tokens must be less than max tokens.")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str):
        tokenized = self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            stride=self.overlap_tokens,
            return_overflowing_tokens=True,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )
    
        input_ids = tokenized["input_ids"]
        attention_masks = tokenized["attention_mask"]

        # Decode each chunk back into text
        decoded_chunks = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in input_ids
        ]

        # normilized_decoded_chunks = [ChunkingStrategy.normilize_text(chunk) for chunk in decoded_chunks]

        return tokenized, decoded_chunks

# Concrete Strategy 5 — Multi-sentence chunking up to a limit
class MultiSentenceChunking(ChunkingStrategy):
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def chunk(self, text: str) -> list[str]:
        sentences = SentenceChunking().chunk(text)
        chunks = []
        current_chunk = ""
        for sentence in tqdm(sentences, total=len(sentences), desc="Chunking", unit="sentence"):
            if len(current_chunk) + len(sentence) + 1 <= self.max_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

# Context — RAG Processor
class RAGProcessor:
    def __init__(self, chunking_strategy: ChunkingStrategy):
        self.chunking_strategy = chunking_strategy

    def process_document(self, text: str):
        chunks = self.chunking_strategy.chunk(text)
        normilized_chunks = []

        if self.chunking_strategy.__class__.__name__ == "TokenLimitedChunking":
            num_chunks = len(chunks['input_ids'])
            print(f"Chunks generated ({num_chunks})")
            for ids in chunks['input_ids']:
                decoded = self.chunking_strategy.tokenizer.decode(ids, skip_special_tokens=True).strip()
                normilized_chunks.append(ChunkingStrategy.normilize_text(decoded))
        else:
            print(f"Chunks generated ({len(chunks)})")
            for chunk in chunks:
                normilized_chunks.append(ChunkingStrategy.normilize_text(chunk))

        return normilized_chunks