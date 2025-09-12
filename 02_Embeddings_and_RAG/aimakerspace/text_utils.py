import os
from typing import List
from pypdf import PdfReader


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
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
            self.documents.append(f.read())

    def load_pdf_file(self):
        try:
            reader = PdfReader(self.path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"
            self.documents.append(pdf_text.strip())
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):
                    try:
                        reader = PdfReader(file_path)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text() + "\n"
                        self.documents.append(pdf_text.strip())
                    except Exception as e:
                        print(f"Warning: Could not read PDF {file_path}: {str(e)}")

    def load_documents(self):
        self.load()
        return self.documents


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
