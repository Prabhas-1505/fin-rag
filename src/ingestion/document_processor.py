import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initializes the document processor with a sliding window chunker.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Using tiktoken for accurate token counting. disallowed_special=()
        # treats token-like substrings (e.g. a document that quotes
        # "<|endoftext|>" as an example attack string) as plain text instead
        # of raising - we're only counting tokens, never executing them.
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            disallowed_special=(),
        )

    def parse_pdf(self, file_path: str) -> str:
        """
        Extracts raw text from a PDF file using PyMuPDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        
        doc = fitz.open(file_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text.append(page.get_text("text"))
        
        return "\n".join(full_text)

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Splits text into chunks using a sliding window approach.
        Returns a list of dictionaries with 'text' and 'metadata'.
        """
        if metadata is None:
            metadata = {}
            
        chunks = self.text_splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "metadata": metadata
            }
            for chunk in chunks
        ]

    def process_pdf_to_chunks(self, file_path: str, source_id: str) -> List[Dict[str, Any]]:
        """
        End-to-end pipeline to read a PDF and return overlapping chunks.
        """
        raw_text = self.parse_pdf(file_path)
        metadata = {"source_id": source_id, "file_path": file_path}
        return self.chunk_text(raw_text, metadata)

if __name__ == "__main__":
    # Quick test if run directly (requires a test.pdf in the same directory)
    processor = DocumentProcessor()
    print("DocumentProcessor initialized.")
