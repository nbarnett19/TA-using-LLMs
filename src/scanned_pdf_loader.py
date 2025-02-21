# scanned_pdf_loader.py

import os
import re
import json
import pandas as pd
from typing import Iterator
from pypdf import PdfReader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract


class ScannedPDFLoader(BaseLoader):
    """A document loader that reads all PDF files in a folder."""

    def __init__(self, folder_path: str) -> None:
        """Initialize the loader with a folder path.

        Args:
            folder_path: The path to the folder containing PDF files.
        """
        self.folder_path = folder_path

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that reads all PDF files in a folder and applies OCR.

        Returns:
            An iterator yielding `Document` objects.
        """
        docs = []
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(self.folder_path, file_name)
                # Convert PDF to images
                pages = convert_from_path(file_path)
                docs = []
                for page_image in pages:
                    # Deskew the image
                    deskewed_image = self.deskew(page_image)

                    # Extract text from the deskewed image
                    doc = self.extract_text_from_image(deskewed_image)
                    docs.append(doc)

        return docs

    def deskew(self, image):
        """Deskew the image for better OCR accuracy."""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.size
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(np.array(image), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def extract_text_from_image(self, image):
        """Extract text from an image using pytesseract."""
        text = pytesseract.image_to_string(image)
        doc = Document(page_content=text, metadata={"source": "local"})
        return doc

    def split_text(self, docs: Optional[List[Document]] = None, chunk_size=1000, chunk_overlap=500) -> List[Document]:
        """Split a list of Document objects into smaller chunks.

        Args:
            docs: A list of Document objects. If not provided, the load method is called to load documents.

        Returns:
            A list of Document objects with smaller chunks.
        """
        # Automatically load documents if they are not provided
        if docs is None:
            docs = self.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

    def semantic_split_text(self, docs: Optional[List[Document]] = None) -> List[Document]:
        """Split a list of Document objects into smaller chunks.

        Args:
            docs: A list of Document objects. If not provided, the load method is called to load documents.

        Returns:
            A list of Document objects with smaller chunks.
        """

        text_splitter = SemanticChunker(OpenAIEmbeddings())
        chunks = text_splitter.split_documents(docs)
        return chunks