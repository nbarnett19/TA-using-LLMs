# chromadb_setup.py

from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from typing import List, Optional

class ChromaVectorStoreManager:
    def __init__(self, collection_name: str, embeddings, persist_directory: str):
        """
        Initializes the Chroma Vector Store Manager.

        Args:
            collection_name (str): The name of the collection to store vectors.
            embeddings: The embedding function to use for vectorization.
            persist_directory (str): Directory to persist the Chroma vector store.
        """
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory

        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )


    def _is_vector_store_empty(self) -> bool:
        """
        Checks if the Chroma vector store is empty.

        Returns:
            bool: True if the vector store is empty, False otherwise.
        """
        try:
            docs = self.vector_store.similarity_search("", k=1)
            return len(docs) == 0  # If no documents are found, the store is empty
        except Exception as e:
            print(f"Error checking vector store: {e}")
            return True

    def _clear_vector_store(self):
        """
        Clears all documents from the Chroma vector store and resets the collection.
        """
        # Delete the existing collection
        self.vector_store.delete_collection()
        print(f"Cleared the collection '{self.collection_name}'.")

        # Re-initialize the vector store after clearing the collection
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document], empty_db: bool = True):
        """
        Adds documents to the Chroma vector store. If the store is not empty, it clears the existing store first.

        Args:
            documents (List[Document]): List of documents to add to the vector store.
            empty_db (bool): If True, the existing vector store will be cleared before adding new documents.
        """
        if empty_db:
            # Check if the vector store is empty
            if not self._is_vector_store_empty():
                # If not empty, clear and reset the vector store
                self._clear_vector_store()

        # Generate unique UUIDs for the new documents
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Add documents to the vector store
        self.vector_store.add_documents(documents=documents, ids=uuids)
        print(f"Added {len(documents)} documents to the collection '{self.collection_name}'.")

    def set_embeddings(self, embeddings):
        """
        Updates the embedding function used by the vector store.

        Args:
            embeddings: The new embedding function to use.
        """
        self.embeddings = embeddings
        # Update vector store with new embedding function if needed
        self.vector_store.embedding_function = embeddings

    def set_collection_name(self, collection_name: str):
        """
        Updates the collection name used by the vector store.

        Args:
            collection_name (str): The new collection name.
        """
        self.collection_name = collection_name
        # Reinitialize vector store with new collection name
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def set_persist_directory(self, persist_directory: str):
        """
        Updates the persist directory where Chroma saves data.

        Args:
            persist_directory (str): New directory to persist the data.
        """
        self.persist_directory = persist_directory
        # Reinitialize vector store with the new directory
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
