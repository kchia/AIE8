import os
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from datetime import datetime


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.metadata = []  # Store metadata for each document
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_file()
            elif self.path.endswith(".pdf"):
                self.load_pdf_file()
            else:
                raise ValueError(
                    "Provided path must be a directory, .txt file, or .pdf file."
                )
        else:
            raise ValueError(
                "Provided path does not exist or is not accessible."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            content = f.read()
            self.documents.append(content)
            # Generate metadata
            metadata = {
                "source_file": os.path.basename(self.path),
                "source_path": self.path,
                "file_type": "txt",
                "document_length": len(content),
                "load_timestamp": datetime.now().isoformat(),
                "page_number": None  # Not applicable for txt files
            }
            self.metadata.append(metadata)

    def load_pdf_file(self):
        try:
            reader = PdfReader(self.path)
            # Option 1: Load entire PDF as one document
            pdf_text = ""
            page_count = len(reader.pages)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                pdf_text += page_text + "\n"
            
            content = pdf_text.strip()
            self.documents.append(content)
            
            # Generate metadata for the entire PDF
            metadata = {
                "source_file": os.path.basename(self.path),
                "source_path": self.path,
                "file_type": "pdf",
                "document_length": len(content),
                "load_timestamp": datetime.now().isoformat(),
                "page_count": page_count,
                "page_range": f"1-{page_count}"
            }
            self.metadata.append(metadata)
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        content = f.read()
                        self.documents.append(content)
                        # Generate metadata
                        metadata = {
                            "source_file": file,
                            "source_path": file_path,
                            "file_type": "txt",
                            "document_length": len(content),
                            "load_timestamp": datetime.now().isoformat(),
                            "page_number": None
                        }
                        self.metadata.append(metadata)
                elif file.endswith(".pdf"):
                    try:
                        reader = PdfReader(file_path)
                        pdf_text = ""
                        page_count = len(reader.pages)
                        for page_num, page in enumerate(reader.pages, 1):
                            page_text = page.extract_text()
                            pdf_text += page_text + "\n"
                        
                        content = pdf_text.strip()
                        self.documents.append(content)
                        
                        # Generate metadata for the entire PDF
                        metadata = {
                            "source_file": file,
                            "source_path": file_path,
                            "file_type": "pdf",
                            "document_length": len(content),
                            "load_timestamp": datetime.now().isoformat(),
                            "page_count": page_count,
                            "page_range": f"1-{page_count}"
                        }
                        self.metadata.append(metadata)
                    except Exception as e:
                        print(f"Warning: Could not read PDF {file_path}: {str(e)}")

    def load_documents(self) -> List[str]:
        self.load()
        return self.documents
    
    def load_documents_with_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load documents and return both documents and their metadata."""
        self.load()
        return self.documents, self.metadata


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    
    def split_texts_with_metadata(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split texts while preserving and enhancing metadata for each chunk."""
        chunks = []
        chunk_metadata = []
        
        for text, original_metadata in zip(texts, metadata_list):
            text_chunks = self.split(text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                
                # Create enhanced metadata for this chunk
                chunk_meta = original_metadata.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_count": len(text_chunks),
                    "chunk_size": len(chunk),
                    "chunk_start_char": i * (self.chunk_size - self.chunk_overlap),
                    "chunk_end_char": i * (self.chunk_size - self.chunk_overlap) + len(chunk)
                })
                chunk_metadata.append(chunk_meta)
                
        return chunks, chunk_metadata


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
