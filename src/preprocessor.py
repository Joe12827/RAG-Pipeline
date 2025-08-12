from abc import ABC, abstractmethod
import pdfplumber
import docx
from bs4 import BeautifulSoup
import unicodedata
import re
from tqdm import tqdm

# Strategy Interface
class DocumentTextExtractor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass

    def clean_text_unicode(text: str) -> str:
        cleaned = []
        for ch in text:
            if ch.isprintable() and unicodedata.category(ch)[0] != "C":
                cleaned.append(ch)
            else:
                cleaned.append(" ")  # replace junk with a space
        # Normalize multiple spaces
        return re.sub(r"\s+", " ", "".join(cleaned)).strip()


# Concrete Strategy 1 — PDF extractor
class PDFTextExtractor(DocumentTextExtractor):
    def extract_text(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page in tqdm(pdf.pages, total=total_pages, desc="Processing PDF", unit="page"):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return DocumentTextExtractor.clean_text_unicode(text)


# Concrete Strategy 2 — DOCX extractor
class DocxTextExtractor(DocumentTextExtractor):
    def extract_text(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        text = "\n".join(paragraphs)
        return DocumentTextExtractor.clean_text_unicode(text)


# Concrete Strategy 3 — Plain text extractor
class PlainTextExtractor(DocumentTextExtractor):
    def extract_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            return DocumentTextExtractor.clean_text_unicode(text)


# Concrete Strategy 4 — HTML extractor
class HTMLTextExtractor(DocumentTextExtractor):
    def extract_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return DocumentTextExtractor.clean_text_unicode(text)


# Context
class DocumentProcessor:
    def __init__(self, extractor: DocumentTextExtractor):
        self.extractor = extractor

    def process(self, file_path: str) -> str:
        return self.extractor.extract_text(file_path)


# Usage example
if __name__ == "__main__":
    max_length = 200  # Limit text length for display purposes

    # PDF example
    pdf_processor = DocumentProcessor(PDFTextExtractor())
    print("PDF Text Extracted:")
    text = pdf_processor.process("Data/sample.pdf")
    print(text[:max_length] + "\n")

    # DOCX example
    docx_processor = DocumentProcessor(DocxTextExtractor())
    print("DOCX Text Extracted:")
    text = docx_processor.process("Data/sample.docx")
    print(text[:max_length] + "\n")

    # TXT example
    txt_processor = DocumentProcessor(PlainTextExtractor())
    print("TXT Text Extracted:")
    text = txt_processor.process("Data/sample.txt")
    print(text[:max_length] + "\n")

    # HTML example
    html_processor = DocumentProcessor(HTMLTextExtractor())
    print("HTML Text Extracted:")
    text = html_processor.process("Data/sample.html")
    print(text[:max_length] + "\n")
