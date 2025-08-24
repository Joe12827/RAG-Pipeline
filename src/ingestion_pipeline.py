from preprocessor import *
from chunker import *
from embedder import *
from document import Document
import os
from dotenv import load_dotenv
from pinecone import Pinecone

class IngestionPipeline:

    def __init__(self, chunker: ChunkingStrategy, embedder: Embedder):
        self.chunker = chunker
        self.embedder = embedder

    def get_preprocessor(self, filename):
        ext = filename.lower().split('.')[-1]
        match ext:
            case 'txt':
                return DocumentProcessor(PlainTextExtractor())
            case 'html':
                return DocumentProcessor(HTMLTextExtractor())
            case 'pdf':
                return DocumentProcessor(PDFTextExtractor())
            case 'docx':
                return DocumentProcessor(DocxTextExtractor())
            case _:
                raise ValueError(f"Unsupported file type: {ext}")
    
    def pinecone_upload(self, documents: list[Document]):
        '''
        Embeddings: List of tuples (file_name, text, vector)
        '''
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("INDEX_NAME")
        pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
        index = pc.Index(INDEX_NAME)

        # Prepare vectors for upsert
        batch_size = 10
        vectors_to_upsert = []
        i = 0
        for doc in documents:
            i += 1
            embed_id = f"chunk-{i}"
            vectors_to_upsert.append((embed_id, doc.vector, doc.get_metadata()))

            if len(vectors_to_upsert) >= batch_size:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)

    def process_documents(self, files):
        """
        Create Document objects from a list of files by processing, chunking, and embedding them.
        """
        # files: List of file paths
        documents = []
        for file_path in files:
            print(f"Processing file: {file_path}")

            try:
                preprocessor = self.get_preprocessor(file_path)
            except ValueError as e:
                print(e)
                continue
            
            file_name = os.path.basename(file_path)
            preprocessed_doc = preprocessor.process(file_path)

            if isinstance(self.chunker, TokenLimitedChunking):
                tokenized_chunks, decoded_chunks = self.chunker.chunk(preprocessed_doc)

                chunk_count = len(tokenized_chunks["input_ids"])
                print(f"Generated {chunk_count} chunks")

                embeddings = self.embedder.embed(tokenized_chunks, already_tokenized=True)

                for embedding, decoded_chunk in zip(embeddings, decoded_chunks):
                    documents.append(Document(vector=embedding, text=decoded_chunk, source=file_name))
            
            else:
                chunks = self.chunker.chunk(preprocessed_doc)

                chunk_count = len(chunks)
                print(f"Generated {chunk_count} chunks")
                
                embeddings = self.embedder.embed(chunks, already_tokenized=False)

                for embedding, chunk in zip(embeddings, chunks):
                    documents.append(Document(vector=embedding, text=chunk, source=file_name))
        
        print(f"Total documents to upload: {len(documents)}")
        for doc in documents[:3]:
            print(doc)
        # self.pinecone_upload(documents)
        