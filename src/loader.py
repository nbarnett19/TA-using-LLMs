# loader.py

from langchain_core.document_loaders import BaseLoader
from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


class FolderLoader(BaseLoader):
    """A document loader that reads all files in a folder."""

    def __init__(self, folder_path: str) -> None:
        """Initialize the loader with a folder path.

        Args:
            folder_path: The path to the folder containing txt files.
        """
        self.folder_path = folder_path

    def load_txt(self) -> List[Document]:
        """Load all txt files in the folder and return a list of Document objects.

        Returns:
            A list of Document objects.
        """
        text_loader_kwargs = {"autodetect_encoding": True}
        loader = DirectoryLoader(
            self.folder_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, show_progress=True
        )
        docs = loader.load()
        doc_sources = [doc.metadata["source"] for doc in docs]
        print(doc_sources)
        return docs

    def load_pdf(self) -> List[Document]:
        """Load all pdf files in the folder and return a list of Document objects.

        Returns:
            A list of Document objects.
        """
        loader = PyPDFDirectoryLoader(self.folder_path)
        docs = loader.load()
        doc_sources = [doc.metadata["source"] for doc in docs]
        print(doc_sources)
        return docs

    def split_text(self, docs: Optional[List[Document]] = None, chunk_size=1000, chunk_overlap=500) -> List[Document]:
        """Split a list of Document objects into smaller chunks.

        Args:
            docs: A list of Document objects. If not provided, the load method is called to load documents.
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between chunks.
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



