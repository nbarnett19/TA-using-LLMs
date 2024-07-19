# transcript_loader.py

import pypdf
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import PyPDFLoader


class TranscriptLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_text_from_pdf(self) -> str:
        """Load text from a PDF file using PyPDFLoader."""
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()  # Load the PDF into a list of Document objects

        # Initialize an empty string to hold the full text
        full_text = ""

        # Extract text from each document
        for doc in documents:
            # Assuming `doc` has a `page_content` attribute which contains the text
            full_text += doc.page_content  # Adjust this if `page_content` is not the correct attribute

        return full_text


