# transcript_loader.py

import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TranscriptLoader:
    """
    Manages the loading and extraction of text from PDF documents.

    Attributes:
        file_path (str): The file path of the PDF document to be loaded.
    """

    def __init__(self, file_path: str):
        """
        Initializes the TranscriptLoader with the given file path.

        Args:
            file_path (str): The file path of the PDF document to be loaded.
        """
        self.file_path = file_path

    def load_text_from_pdf(self) -> str:
        """
        Load text from a PDF file using PyPDFLoader.

        This method uses the PyPDFLoader from the LangChain library to load the content of a PDF file.
        It then concatenates the text content of each page into a single string.

        Returns:
            str: The full text extracted from the PDF document.
        """
        loader = PyPDFLoader(self.file_path)
        data = loader.load()  # Load the PDF into a list of Document objects

        # Initialize an empty string to hold the full text
        text = ""

        # Extract text from each document
        for doc in data:
            # Assuming `doc` has a `page_content` attribute which contains the text
            text += doc.page_content
        return text

    def split_text_into_chunks(self, chunk_size=5000, chunk_overlap=1000, separators=['.']):
        """
        Split text into chunks with overlap.

        This method uses the RecursiveCharacterTextSplitter from the LangChain library to split text.
        It ovelaps the text based on the chunk_overlap parameter.

        Args:
            chunk_size (int): The maximum length of each chunk.
            chunk_overlap (int): The overlap between chunks

        Returns:
            str: The chunks extracted from the PDF document.
        """
        rc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

        # Load extracted text
        text = self.load_text_from_pdf()

        # Split text into chunks
        chunks = rc_splitter.split_text(text)

        return chunks



