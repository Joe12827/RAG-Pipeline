import spacy
from abc import ABC, abstractmethod
from tqdm import tqdm


# Strategy Interface
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


# Concrete Strategy 1 — Fixed-size character chunks
class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]


# Concrete Strategy 2 — Sentence-based chunking
class SentenceChunking(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences


# Concrete Strategy 3 — Paragraph-based chunking
class ParagraphChunking(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


# Concrete Strategy 4 — Token-limited chunking
class TokenLimitedChunking:
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
        return self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            stride=self.overlap_tokens,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False
        )

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

        if self.chunking_strategy.__class__.__name__ == "TokenLimitedChunking":
            num_chunks = len(chunks['input_ids'])
            print(f"Chunks generated ({num_chunks})")
        else:
            print(f"Chunks generated ({len(chunks)})")

        return chunks