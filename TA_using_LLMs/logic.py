#TA_using_LLMs\logic.py
import os
import getpass
import re
import json
import langchain
import langchain_core
import langchain_community
import pandas as pd
import nltk
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datasets
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from nltk import ngrams as nltk_ngrams
from typing import List, Optional, Any, Dict, Tuple
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import Iterator
from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
# from langchain_core.retrievers import BaseRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fuzzywuzzy import fuzz
from multiprocessing import Pool
from collections import Counter
from tqdm.auto import tqdm
from typing import Optional, List, Tuple
from datasets import Dataset
from huggingface_hub import InferenceClient, notebook_login
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
nltk.download('punkt')


class ModelManager:
    """
    Manages the initialization and configuration of different language models.

    Attributes:
        model_choice (str): The choice of model to initialize.
        temperature (float): Temperature for controlling randomness in text generation.
        top_p (float): Nucleus sampling parameter for controlling diversity in text generation.
        llm: The initialized language model.
    """
    def __init__(self, model_choice='gemini-1.5-flash', temperature=0.5, top_p=0.5):
        """
        Initializes the ModelManager with the given model choice, temperature, and top_p settings.

        Args:
            model_choice (str): The choice of model to initialize.
            temperature (float): Temperature for controlling randomness in text generation.
            top_p (float): Nucleus sampling parameter for controlling diversity in text generation.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Ensure API keys are available
        self._ensure_api_keys()
        self.model_choice = model_choice
        self.temperature = temperature
        self.top_p = top_p
        self.llm = self._initialize_model(model_choice, temperature, top_p)

    def _ensure_api_keys(self):
        """
        Prompts the user to enter API keys if they are not set in environment variables.
        """
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")

        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OpenAI API Key: ")

    def _initialize_model(self, model_choice, temperature, top_p):
        """
        Initializes the appropriate language model based on the model choice.

        Args:
            model_choice (str): The choice of model to initialize.
            temperature (float): The temperature setting for text generation.
            top_p (float): The top_p setting for nucleus sampling.

        Returns:
            An instance of the chosen language model.

        Raises:
            ValueError: If an unknown model choice is provided.
        """
        if model_choice.startswith('gemini'):
            gemini_api_key = os.getenv('GOOGLE_API_KEY')
            if not gemini_api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set in environment variables")
            return ChatGoogleGenerativeAI(model=model_choice, temperature=temperature, google_api_key=gemini_api_key, top_p=top_p)
        elif model_choice.startswith('gpt'):
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
            return ChatOpenAI(model=model_choice, temperature=temperature, api_key=openai_api_key, top_p=top_p)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")

    def update_parameters(self, temperature=None, top_p=None):
        """
        Updates the temperature and top_p parameters for the language model.

        Args:
            temperature (float, optional): New temperature setting.
            top_p (float, optional): New top_p setting.
        """
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p

        # Reinitialize the model with the updated parameters
        self.llm = self._initialize_model(self.model_choice, self.temperature, self.top_p)


class FocusGroup(BaseModel):
    focus_group: Optional[int] = Field(description="The focus group number")
    date: Optional[str] = Field(description="The date of the focus group")
    participants: Optional[List[str]] = Field(description="The participants of the focus group")
    content: str = Field(description="The content of the focus group excluding lines with focus group number and date")


class CodeExcerpt(BaseModel):
    code: str = Field(description="The code or theme")
    code_description: str = Field(description="The description of the code")
    excerpt: str = Field(description="The excerpt supporting the code")
    speaker: Optional[str] = Field(description="The speaker of the line")


class Themes(BaseModel):
    theme: str = Field(description="The themes of the text")
    theme_definition: str = Field(description="The definition of the themes")
    subthemes: List[str] = Field(description="The subthemes of the theme")
    subtheme_definitions: List[str] = Field(description="The definitions of the subthemes")
    supporting_quotes: List[str] = Field(description="The supporting quotes for the theme or subthemes")


class ZSControl(BaseModel):
    theme: str = Field(description="The theme of the text")
    theme_definition: str = Field(description="The definition of the theme")
    subthemes: List[str] = Field(description="The subthemes of the theme")
    subtheme_definitions: List[str] = Field(description="The definitions of the subthemes")
    codes: List[str] = Field(description="The supporting codes for the theme or subthemes")
    supporting_quotes: str = Field(description="The excerpt supporting the code")
    speaker: str = Field(description="The speaker of the line denoted by TN")


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

    def split_text(self, docs=None, chunk_size=1000, chunk_overlap=500):
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

    def semantic_split_text(self, docs=None):
        """Split a list of Document objects into smaller chunks.

        Args:
            docs: A list of Document objects. If not provided, the load method is called to load documents.

        Returns:
            A list of Document objects with smaller chunks.
        """

        text_splitter = SemanticChunker(OpenAIEmbeddings())
        chunks = text_splitter.split_documents(docs)
        return chunks


class ThematicAnalysis:
    """
    Generates themes with definitions and supporting quotes from the text.

    Attributes:
        llm: The language model used to generate responses.
        docs (list):The full documents to analyze.
        chunks (list): The text chunks to analyze.
        rqs (str): The research questions to answer.
    """
    def __init__(self, llm, docs, chunks, rqs):
        self.llm = llm
        self.docs = docs
        self.chunks = chunks
        self.rqs = rqs

    def generate_summary(self):
        """
        Generates summary from the text based on research questions.

        Returns:
            The generated response from the language model for the combined text.
        """
        template = """You are a qualitative researcher. The aim of your study is
        to answer the following research questions: {rqs}
        Based on your research questions, generate a short summary from the text: {text}."""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm

        summaries = []
        for chunk in self.docs:
            try:
                # Generate summary for each chunk
                summary = chain.invoke({"rqs": self.rqs, "text": chunk.page_content})
                summaries.append(summary.content)
            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        # Combine all chunk summaries into a single summary
        combined_summary_template = """You are a qualitative researcher.
        The aim of your study is to answer the following research questions: {rqs}
        Based on your research questions and the following summaries,
        generate an overall summary: {summaries}."""
        combined_summary_prompt = PromptTemplate.from_template(combined_summary_template)
        combined_chain = combined_summary_prompt | self.llm

        try:
            final_summary = combined_chain.invoke({"rqs": self.rqs, "summaries": "\n\n".join(summaries)})
            print(final_summary.usage_metadata)
            print("Final summary:", final_summary.content)
            return final_summary.content
        except Exception as e:
            print(f"Error occurred while generating final summary: {e}")
            raise

    def zs_control_gemini(self, filename=None) -> Any:
        """
        Generates themes with definitions, subthemes with definitions, codes and excerpts.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """

        prompt_template = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Based on the research questions provided, you must identify a maximum of 6 distinct themes.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The transcripts: {text}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["rqs", "format_instructions", "text"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        # Initialize an empty str
        all_text = ""

        # Iterate through each focus group in the data
        for item in self.docs:
            # Iterate through the content of the focus group
            for text in item.page_content:
              # Append each text to the text_chunk string
              all_text += text

        try:
          results = chain.invoke({
              "rqs": self.rqs,
              "text": all_text
              })

          print(prompt.template.format(
                rqs=self.rqs,
                text=all_text,
                format_instructions=format_instructions
            ))

          if filename:
              if filename.endswith('.json'):
                  with open(filename, 'w') as f:
                      json.dump(results, f, indent=4)
                      print(f"Results successfully saved to {filename}")
              elif filename.endswith('.csv'):
                  df = pd.DataFrame(results)
                  df.to_csv(filename, index=False)
                  print(f"Results successfully saved to {filename}")
              else:
                  print("Invalid file format. Please use .json or .csv.")

          return results

        except Exception as e:
            print(f"Error occurred while processing: {e}")
            raise

    def zs_control_gpt(self, filename=None) -> Any:
        """
        Generates themes with definitions, subthemes with definitions, codes, and excerpts.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            A single JSON object containing all themes, sub-themes, and codes across all chunks.
        """

        prompt_template = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Based on the research questions provided, you must identify a maximum of 6 distinct themes.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The transcripts: {text}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["rqs", "format_instructions", "text"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_themes = []

        for data in self.docs:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing file: {source_file}")

                # Extract text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }

                # Generate themes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a list, use it directly
                if isinstance(response, list):
                    themes = response
                else:
                    # If the response is a single dictionary, wrap it in a list
                    if isinstance(response, dict):
                        themes = [response]
                    else:
                        raise ValueError("Unexpected format: response must be a list or dictionary.")

                # Ensure themes are in the expected format
                for theme in themes:
                    if not isinstance(theme, dict) or not all(key in theme for key in ["theme", "theme_definition", "subthemes", "subtheme_definitions", "codes", "supporting_quotes", "speaker"]):
                        raise ValueError("Invalid theme format detected.")

                    # Add focus group information
                    theme['source file'] = source_file

                # Flatten the results
                all_themes.extend(themes)

            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        # Second prompt for refining and filtering themes
        prompt_template2 = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Provided are a list previously
        identified themes from the data. Review the given list of themes and combine or
        filter them to identify a maximum of 6 distinct themes that address the research questions.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The list of themes: {themes}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template2,
            input_variables=["rqs", "format_instructions", "themes"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        # Prepare input dictionary
        input_data = {
            "rqs": self.rqs,
            "themes": all_themes
        }

        try:
          final_themes = chain.invoke(input_data)
          print("Final output:", final_themes)
        except Exception as e:
          print(f"Error occurred while generating final themes: {e}")
          raise

        print(prompt.template.format(
          rqs=self.rqs,
          themes=all_themes,
          format_instructions=format_instructions
      ))

        # Optionally save to a file
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(final_themes, f, indent=4)
                    print(f"Results successfully saved to {filename}")
            elif filename.endswith('.csv'):
                df = pd.DataFrame(final_themes)
                df.to_csv(filename, index=False)
                print(f"Results successfully saved to {filename}")
            else:
                print("Invalid file format. Please use .json or .csv.")

        return final_themes


class GenerateCodes(ThematicAnalysis):
    """
    Generates codes and supporting quotes from the text.

    Attributes:
        llm: The language model used to generate responses.
        chunks (str): The text chunks to analyze.
        rqs (str): The research questions to answer.
    """
    def __init__(self, llm, docs, chunks, rqs, examples=None, vector_db=None, retriever=None):
        super().__init__(llm, docs, chunks, rqs)
        self.examples = examples
        self.vector_db = vector_db
        self.retriever = retriever

    def query_transformation(self, template: str, questions: str) -> list:
        """Generates sub-questions from the given research question using decomposition."""

        prompt_transformation = ChatPromptTemplate.from_template(template)
        generate_queries_transformation = (
            prompt_transformation | self.llm | StrOutputParser()
        )

        # Invoke the decomposition chain
        sub_questions = generate_queries_transformation.invoke({"questions": questions})
        return sub_questions

    def generate_codes(self, filename: Optional[str] = None, use_rag: bool = False,
                       rag_query: Optional[str] = None, similarity_search_with_score: bool = False) -> pd.DataFrame:
        """
        Generates codes and supporting quotes from the text, with optional RAG.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.
            use_rag (bool): If True, use RAG to fetch relevant documents before generating codes.
            rag_query (Optional[str]): Optional query to use for RAG.
            similarity_search_with_score (bool): If True, use similarity search with score.
        """

        prompt_codes_template = """You are a qualitative researcher and are doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Follow this format {format_instructions}
        Research questions: {rqs}
        The transcripts: {text}
        """
        # Add examples to the template if provided
        if self.examples:
            prompt_codes_template += "Examples: {examples}"

        if use_rag:
            prompt_codes_template = "Context: {context}\n" + prompt_codes_template

        # Ensure JsonOutputParser and format_instructions are correctly defined
        parser = JsonOutputParser(pydantic_object=CodeExcerpt)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_codes_template,
            input_variables=["text", "rqs", "format_instructions", "examples", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_codes = []
        counter = 1

        for data in self.chunks:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing chunk {counter}")
                counter += 1

                # Extract Text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }
                if self.examples:
                    input_data["examples"] = self.examples

                # If RAG is enabled, retrieve relevant documents from the Chroma vector database
                retrieved_docs = []
                if use_rag:
                    original_prompt = prompt.template.format(
                                text=data,
                                rqs=self.rqs,
                                format_instructions=format_instructions,
                                examples=self.examples if self.examples else "",
                                context="")
                    rta_questions = """ How does one perform inductive (latent/semantic)
                    reflexive Thematic analysis according to the book practical guide
                    from Braun and Clark (2022)? How can one identify excerpts (or quotes)
                    that address the research questions in reflexive thematic analysis? """
                    if rag_query is not None:
                        # Decompose research questions into sub-questions
                        sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions + text)
                        results = self.retriever.invoke(sub_questions)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                    elif similarity_search_with_score==True:
                        results = self.vector_db.similarity_search_with_score(original_prompt)
                        for doc, score in results:
                           retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                    else:
                        results = self.retriever.invoke(original_prompt)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

                    # Combine the retrieved documents with the original chunk text
                    if similarity_search_with_score:
                      input_data["context"] = "\n".join([doc.page_content for doc, _ in results])
                    else:
                      input_data["context"] = "\n".join([doc.page_content for doc in results])
                    print(f"Retrieved documents: {retrieved_docs}")

                # Generate codes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a single dictionary, convert it to a list of one item
                if isinstance(response, dict):
                  response = [response]

                # If response is a list, use it directly
                if isinstance(response, list):
                  codes = response

                  # Ensure codes are in the expected format
                  for code in codes:
                      if not isinstance(code, dict) or not all(key in code for key in ["code", "excerpt", "speaker"]):
                          raise ValueError("Invalid code format detected.")

                      # Fill missing values
                      code['chunk_analyzed'] = text
                      code['source'] = source_file  # Add source file information
                      if rag_query is not None:
                          code['RAG_query'] = sub_questions
                          code['retrieved_documents'] = retrieved_docs
                      elif use_rag:
                          code['retrieved_documents'] = retrieved_docs
                      else:
                          continue

                  all_codes.extend(codes)  # Flatten the results
                else:
                    raise ValueError("Unexpected response format from model.")

            except Exception as e:
                print(f"Error occurred while processing chunk {counter} in {source_file}: {e}")

        try:
            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(all_codes)
            print(f"DataFrame shape: {df.shape}")
            print(prompt.template.format(
                            text=text,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context=retrieved_docs if use_rag else ""
                        ))  # Add the prompt
            if rag_query is not None:
                print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(all_codes, f, indent=4)
                        print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return all_codes

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")
            raise

    def cot_coding(self, filename: Optional[str] = None, use_rag: bool = False,
                   rag_query: Optional[str] = None, similarity_search_with_score: bool = False):
        """
        Generates codes and supporting quotes from the text.

        """
        cot_prompt_template = """
        You are a qualitative researcher and are doing inductive (latent/semantic)
        reflexive Thematic analysis according to the book practical guide from
        Braun and Clark (2022).
        Follow these steps to analyze the transcripts:

        1. **Review the Transcripts:** Carefully read the transcripts to identify
        key excerpts related to the research questions.

        2. **Generate Codes:** Generate codes that best represent each of the excerpts identified.
        Each code should represent the meaning in the excerpt. Codes should be a mix of
        semantic and latent codes. Semantic means the analysis captures meanings
        that are explicitly stated in the data, so that words themselves are taken at face value.
        Latent means the analysis captures meanings not explicitly stated in the data,
        including the ideas, assumptions, or concepts that underpin what is explicitly stated.

        3. **Match Excerpts:** For each code, find exact excerpts from the transcripts that support it.

        4. **Describe the code:** For each code, describe its meaning in the excerpt.

        4. **Organize the Results:** Format your findings according to the following: {format_instructions}.

        Example Research Questions:
        What are educators’ general attitudes toward the promotion of student wellbeing
        and towards a set of ‘wellbeing guidelines’ recently introduced in Irish
        post-primary schools. What are the potential barriers to wellbeing promotion
        and what are educators’ opinions as to what might constitute opposite remedial
        measures in this regard?

        **Example Transcript:**
        P1: I think anything that you do in school that's on paper is difficult
        to relate to students. And, this is the great thing about the new junior-cycle,
        there's a lot more of the hands on approach in most academic subjects.
        I think, that needs to be brought into areas like SPHE. Theory is fine -
        I don't know if you want me to talk about the wellbeing indicators
        [interviewer gestures to continue]. I have them there on my wall, this is
        maybe my third year to have them on my wall. To be honest, I feel that that's
        just way too abstract!
        P2: Although the hands-on approach simulates real life scenarios, I find
        that thoroughly teaching the theory provides students with the tools they
        need to be successful. And some of my students prefer this approach.

        **Example Output:**
        Code name: The wellbeing curriculum is not relatable for the students
        Code description: The participant noted the difficulty students have in relating to school curricula.,
        Excerpt: I think that anything that you do in school that's on paper is difficult to relate to students.

        Code name: A practical approach to learning is beneficial for students
        Code description: The participant praised the hands on approach in the new junior-cycle
        Excerpt: And, this is the great thing about the new junior-cycle, there's
        a lot more of the hands on approach in most academic subjects.

        Code name: Wellbeing promotion should be practical
        Code description: The participant felt there is a need to bring practical approaches into SPHE
        Excerpt: I think, that needs to be brought into areas like SPHE.

        Code name: The wellbeing guidelines lack clarity!
        Code description: The participant emphasized how abstract they found the current written guidelines.
        Excerpt: To be honest, I feel that that's just way too abstract!

        Code name: Theoretical approach is necessary for wellbeing promotion success
        Code description: Participant preferred the theoretical approach to well-being
        education as it fit some students’ learning styles.
        Excerpt: I find that thoroughly teaching the theory provides students with the tools they need to be successful.

        Now, apply this process to the provided transcripts.
        transcript: {text}
        research questions: {rqs}
        """

        if use_rag:
            cot_prompt_template = "Context: {context}\n" + cot_prompt_template

        # Ensure JsonOutputParser and format_instructions are correctly defined
        parser = JsonOutputParser(pydantic_object=CodeExcerpt)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=cot_prompt_template,
            input_variables=["text", "rqs", "format_instructions", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_codes = []
        counter = 1

        for data in self.chunks:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing chunk {counter}")
                counter += 1

                # Extract Text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }

                # If RAG is enabled, retrieve relevant documents from the Chroma vector database
                retrieved_docs = []
                if use_rag:
                    original_prompt = prompt.template.format(
                                text=data,
                                rqs=self.rqs,
                                format_instructions=format_instructions,
                                examples=self.examples if self.examples else "",
                                context="")
                    rta_questions = """ How does one perform inductive (latent/semantic)
                    reflexive Thematic analysis according to the book practical guide
                    from Braun and Clark (2022)? How can one identify excerpts (or quotes)
                    that address the research questions in reflexive thematic analysis? """
                    if rag_query is not None:
                        # Decompose research questions into sub-questions
                        sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions + text)
                        results = self.retriever.invoke(sub_questions)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                    elif similarity_search_with_score==True:
                        results = self.vector_db.similarity_search_with_score(original_prompt)
                        for doc, score in results:
                           retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                    else:
                        results = self.retriever.invoke(original_prompt)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

                    # Combine the retrieved documents with the original chunk text
                    if similarity_search_with_score:
                      input_data["context"] = "\n".join([doc.page_content for doc, _ in results])
                    else:
                      input_data["context"] = "\n".join([doc.page_content for doc in results])
                    print(f"Retrieved documents: {retrieved_docs}")

                # Generate codes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a single dictionary, convert it to a list of one item
                if isinstance(response, dict):
                  response = [response]

                # If response is a list, use it directly
                if isinstance(response, list):
                  codes = response

                  # Ensure codes are in the expected format
                  for code in codes:
                      if not isinstance(code, dict) or not all(key in code for key in ["code", "excerpt", "speaker"]):
                          raise ValueError("Invalid code format detected.")

                      # Fill missing values
                      code['chunk_analyzed'] = text
                      code['source'] = source_file  # Add source file information
                      if rag_query is not None:
                          code['RAG_query'] = sub_questions
                          code['retrieved_documents'] = retrieved_docs
                      elif use_rag:
                          code['retrieved_documents'] = retrieved_docs
                      else:
                          continue

                  all_codes.extend(codes)  # Flatten the results
                else:
                    raise ValueError("Unexpected response format from model.")

            except Exception as e:
                print(f"Error occurred while processing chunk {counter} in {source_file}: {e}")

        try:
            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(all_codes)
            print(f"DataFrame shape: {df.shape}")
            print(prompt.template.format(
                            text=text,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            context=retrieved_docs if use_rag else ""
                        ))  # Add the prompt
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(all_codes, f, indent=4)
                        print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return all_codes

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")
            raise


class GenerateThemes:
    """
    Generates themes and supporting quotes from the codes list.

    Attributes:
        llm: The language model used to generate responses.
        rqs (str): The research questions to answer.
        json_codes_list (List[Dict[str, Any]]): List of codes containing generated codes and their details.
        examples (Optional[List[str]]): Optional list of examples to include in the prompt.
    """
    def __init__(self, llm, rqs, json_codes_list, examples=None, vector_db=None, retriever=None):
        self.llm = llm
        self.rqs = rqs
        self.json_codes_list = json_codes_list
        self.examples = examples
        self.vector_db = vector_db
        self.retriever = retriever

    def query_transformation(self, template: str, questions: str) -> list:
        """Generates sub-questions from the given research question using decomposition."""

        prompt_transformation = ChatPromptTemplate.from_template(template)
        generate_queries_transformation = (
            prompt_transformation | self.llm | StrOutputParser()
        )

        # Invoke the decomposition chain
        sub_questions = generate_queries_transformation.invoke({"questions": questions})
        return sub_questions

    def generate_themes(self, filename = None, use_rag: bool = False,
                        rag_query = None, similarity_search_with_score: bool = False):
        """
        Generates themes with definitions and supporting quotes from the codes list.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """
        prompt_codes_template = """You are a qualitative researcher and are doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022).
        Based on the research questions provided, you need to collate the codes
        into a maximum of 6 distinct themes. Each theme should include:
        1. A theme definition including a title
        2. Sub-themes if needed
        3. Each sub-theme should have a definition
        4. 2 supporting quotes for each theme and subtheme
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.

        Codes: {codes}
        Follow this format: {format_instructions}.
        Research questions: {rqs} """

        # Add examples to the template if provided
        if self.examples:
            prompt_codes_template += "Examples: {examples}"
        # Add context if use_rag = true
        if use_rag:
            prompt_codes_template = "Context: {context}" + prompt_codes_template

        parser = JsonOutputParser(pydantic_object=Themes)
        format_instructions = parser.get_format_instructions()

        # Filter json data to only necessary fields
        fields_to_keep = ["code", "code_description", "excerpt"]

        # Create a new list to store filtered data
        filtered_data = []
        for item in self.json_codes_list:
            filtered_item = {field: item.get(field) for field in fields_to_keep} # Extract specified fields from each dictionary
            filtered_data.append(filtered_item)

        prompt = PromptTemplate(
            template=prompt_codes_template,
            input_variables=["codes", "rqs", "format_instructions", "examples", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        try:
            # Prepare input dictionary
            input_data = {
              "codes": filtered_data,
              "rqs": self.rqs
              }
              # Add examples to the template if provided
            if self.examples:
                input_data["examples"] = self.examples

            # If RAG is enabled, retrieve relevant documents from the Chroma vector database
            retrieved_docs = []
            if use_rag:
                original_prompt = prompt.template.format(
                            codes=filtered_data,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context="")
                rta_questions = """ How does one perform inductive (latent/semantic)
                reflexive Thematic analysis according to the book practical guide
                from Braun and Clark (2022)? How can one collate codes into themes
                that address the research questions in reflexive thematic analysis?
                When defining the themes and subthemes, what does it mean to look for data
                (codes, quotations) that contradict or are discrepant to the established
                themes and subthemes? """
                if rag_query is not None:
                    # Decompose research questions into sub-questions
                    sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions)
                    results = self.retriever.invoke(sub_questions)
                    for doc in results:
                        retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                elif similarity_search_with_score==True:
                    results = self.vector_db.similarity_search_with_score(original_prompt)
                    for doc, score in results:
                        retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                else:
                    results = self.retriever.invoke(original_prompt)
                    for doc in results:
                        retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

            # Combine the retrieved documents with the original chunk text
            if use_rag:
              input_data["context"] = retrieved_docs

            # Generate themes
            results = chain.invoke(input_data)
            print(prompt.template.format(
                    codes=filtered_data,
                    rqs=self.rqs,
                    format_instructions=format_instructions,
                    examples=self.examples if self.examples else "",
                    context=retrieved_docs if use_rag else ""
                ))
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=4)
                    print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df = pd.DataFrame(results)
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return results

        except Exception as e:
            print(f"Error occurred while processing themes: {e}")
            raise

    def cot_themes(self, filename = None, use_rag: bool = False,
                   rag_query = None, similarity_search_with_score: bool = False):
        """
        Generates themes with definitions and supporting quotes from the codes list.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """
        cot_theme_template = """
        Objective: You are a qualitative researcher and are doing inductive
        (latent/semantic) reflexive Thematic analysis according to the book practical
        guide from Braun and Clark (2022).

        Steps:
        1. Group codes into subthemes: Organize related codes into subthemes, if needed, that
        capture shared meanings across the codes based on the research questions provided.
        When subthemes are present, provide a definition for each subtheme.

        2. Group subthemes into themes: Organize related subthemes (if present) or codes into a
        maximum of 6 distinct themes that capture shared meanings across the subthemes
        based on the research questions provided. A subtheme sits under a theme.
        It focuses on one particular aspect of that theme; it brings analytic attention
        and emphasis on this aspect. Use subthemes only when they are needed to
        bring emphasis to one particular aspect of a theme.
                                                                                                                                                                                                                                                                               Support each theme and subtheme (if needed) with at least 2 supporting quotes.
        3. Provide a clear definition for each theme, showing how it addresses
        the research questions. In case you have subthemes do the same with these (definition).
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.

        4. Present Findings: Use this format: {format_instructions}.

        ### Example Analysis:

        **Example Research Questions:**
        What are educators’ general attitudes toward the promotion of student wellbeing
        and towards a set of ‘wellbeing guidelines’ recently introduced in Irish
        post-primary schools. What are the potential barriers to wellbeing promotion
        and what are educators’ opinions as to what might constitute opposite remedial measures in this regard?

        **Example Codes:**
        Code name: The wellbeing curriculum is not relatable for the students
        Code description: The participant noted the difficulty students have in relating to school curricula.

        Code name: A practical approach to learning is beneficial for students
        Code description: The participant praised the hands on approach in the new junior-cycle

        Code name: Wellbeing promotion should be practical
        Code description: The participant felt there is a need to bring practical approaches into SPHE

        Code name: The wellbeing guidelines lack clarity
        Code description": The participant emphasized how abstract they found the current guidelines.

        Code name: Wellbeing promotion requires involvement from all staff members
        Code description: The participant stressed that effective wellbeing promotion
        demands active participation from all staff members, not just a select few.

        Code name: Staff collaboration enhances student wellbeing outcomes
        Code description: The participant highlighted the importance of collaboration
        among school staff in ensuring positive wellbeing outcomes for students.

        Code name: School leadership plays a crucial role in driving wellbeing initiatives
        Code description: The participant emphasized that school leadership is
        key to implementing and sustaining successful wellbeing promotion efforts.

        Code name: Theoretical approach is necessary for wellbeing promotion success
        Code description: Participants preferred the theoretical approach to well-being
        education as it fit some students’ learning styles.

        **Example Output:**
        Theme: An integrative approach to wellbeing promotion
        Theme definition: This theme captures two distinct yet complementary approaches
        to enhancing wellbeing promotion within schools. One narrative emphasizes
        the collective responsibility of the entire school staff in fostering student
        wellbeing, while the other focuses on taking students learning preferences into
        account with the majority of  students preferring a practical, hands-on approach
        for effective wellbeing promotion. Together, these sub-themes represent two
        independently valuable perspectives on how best practices can be applied to
        create meaningful, actionable outcomes in wellbeing initiatives.

        Subthemes:
        Subtheme: Taking student learning preferences into account with the delivery of wellbeing promotion
        Subtheme definition: Many participants highlighted the need for practical wellbeing
        promotion, however, there were some discrepant opinions which suggest a theoretical
        base is still considered necessary
        Relevant codes: A practical approach to learning is beneficial for students,
        Wellbeing promotion should be practical, The wellbeing guidelines lack clarity,
        The wellbeing curriculum is not relatable for the students, Theoretical approach is
        necessary for wellbeing promotion success

        Subtheme: The Whole-School Approach
        Subtheme definition: This subtheme emphasizes the importance of involving
        all members of the school community in promoting student wellbeing. Participants
        stressed that wellbeing should not be confined to a single department or role,
        but rather integrated throughout the entire school.
        Relevant codes: Wellbeing promotion requires involvement from all staff members,
        Staff collaboration enhances student wellbeing outcomes, School leadership
        plays a crucial role in driving wellbeing initiatives

        Now, apply this process to the provided codes, ensuring that each step is followed meticulously.
        Your final output should include a maximum list of 6 themes and subthemes
        if needed, each with their respective definitions and supporting quotes
        that accurately reflect the data.

        codes: {codes}
        research questions: {rqs}
        """
        # Add context if use_rag = true
        if use_rag:
            cot_theme_template = "Context: {context}" + cot_theme_template

        parser = JsonOutputParser(pydantic_object=Themes)
        format_instructions = parser.get_format_instructions()

        # Filter json data to only necessary fields
        fields_to_keep = ["code", "code_description", "excerpt"]

        # Create a new list to store filtered data
        filtered_data = []
        for item in self.json_codes_list:
            filtered_item = {field: item.get(field) for field in fields_to_keep} # Extract specified fields from each dictionary
            filtered_data.append(filtered_item)

        prompt = PromptTemplate(
            template=cot_theme_template,
            input_variables=["codes", "rqs", "format_instructions"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        try:
            # Prepare input dictionary
            input_data = {
              "codes": filtered_data,
              "rqs": self.rqs
              }

            # If RAG is enabled, retrieve relevant documents from the Chroma vector database
            retrieved_docs = []
            if use_rag:
                original_prompt = prompt.template.format(
                            codes=filtered_data,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context="")
                rta_questions = """ How does one perform inductive (latent/semantic)
                reflexive Thematic analysis according to the book practical guide
                from Braun and Clark (2022)? How can one collate codes into themes
                that address the research questions in reflexive thematic analysis?
                When defining the themes and subthemes, what does it mean to look for data
                (codes, quotations) that contradict or are discrepant to the established
                themes and subthemes? """
                if rag_query is not None:
                    # Decompose research questions into sub-questions
                    sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions)
                    results = self.retriever.invoke(sub_questions)
                    for doc in results:
                        retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                elif similarity_search_with_score==True:
                    results = self.vector_db.similarity_search_with_score(original_prompt)
                    for doc, score in results:
                        retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                else:
                    results = self.retriever.invoke(original_prompt)
                    for doc in results:
                        retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

            # Combine the retrieved documents with the original chunk text
            if use_rag:
              input_data["context"] = retrieved_docs

            # Generate themes
            results = chain.invoke(input_data)
            print(prompt.template.format(
                    codes=filtered_data,
                    rqs=self.rqs,
                    format_instructions=format_instructions,
                    context=retrieved_docs if use_rag else ""
                ))
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=4)
                    print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df = pd.DataFrame(results)
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return results

        except Exception as e:
            print(f"Error occurred while processing themes: {e}")
            raise

class QuoteMatcher:
    def __init__(self, docs, chunks, json_codes_list=None, themes_list=None):
        """
        Initializes the QuoteMatcher with the JSON list and JSON codes list.

        Args:
            docs (List[Dict[str, Any]]):The list of documents to analyze.
            chunks (List[Dict[str, Any]]): The list of text chunks to analyze.
            json_codes_list (List[Dict[str, Any]]): The JSON codes list containing the quotes for matching.
            themes_list (List[Dict[str, Any]]): The JSON themes list containing the quotes for matching.
        """
        self.docs = docs
        self.chunks = chunks
        self.json_codes_list = json_codes_list
        self.themes_list = themes_list

    def matched_theme_quotes(self, threshold=80) -> List[Dict]:
        """
        Finds matched quotes from the list of JSON dictionaries against the content in docs.

        Args:
            threshold (int): The similarity threshold for fuzzy matching.

        Returns:
            List[Dict]: A list of dictionaries with unmatched quotes and their indices.
        """
        quotes = []

        # Check if themes_list is a single dictionary, wrap it in a list
        if isinstance(self.themes_list, dict):
            themes = [self.themes_list]  # Single theme, wrap in list
        else:
            themes = self.themes_list  # Multiple themes

        # Put quotes from themes_list in a list
        for item in themes:
            for quote in item["supporting_quotes"]:
                quotes.append(quote)

        # Total quotes that need to be matched
        print(f"Total number of quotes: {len(quotes)}")

        # Match quotes to chunks
        results = []
        for item in quotes:
            highest_match = None
            highest_ratio = 0

            for chunk in self.chunks:
                match_ratio = fuzz.partial_ratio(item, chunk.page_content)

                if match_ratio > highest_ratio:
                    highest_ratio = match_ratio
                    highest_match = {
                        "quote": item,
                        "matched_chunk": chunk.page_content,
                        "match_ratio": match_ratio,
                        "chunk_id": chunk.metadata.get("source", "unknown")
                    }

                if match_ratio >= threshold:
                    # If the match is above the threshold, add it and stop looking for better matches
                    results.append(highest_match)
                    break
            else:
                # If no match was found above the threshold, add the highest match
                if highest_match:
                    results.append(highest_match)

        print(pd.json_normalize(results))

        return results

    def unmatched_code_excerpts(self, threshold=80):
        """
        Identifies excerpts that do not sufficiently match their corresponding chunk_analyzed fields.

        Args:
            threshold (int): The minimum similarity score required to consider a match (default is 50).

        Returns:
            list of dict: A list of dictionaries with unmatched excerpts, chunks, similarity scores, and their indices.
        """
        unmatched_results = []

        for index, item in enumerate(self.json_codes_list):
            excerpt = item.get("excerpt", "")
            chunk_analyzed = item.get("chunk_analyzed", "")
            score = fuzz.partial_ratio(chunk_analyzed, excerpt)

            if score < threshold:
                unmatched_results.append({
                    "index": index,
                    "code": item.get("code", ""),
                    "excerpt": excerpt,
                    "chunk_analyzed": chunk_analyzed,
                    "match_ratio": score
                })

        if unmatched_results is None:
          print("No unmatched results found.")
        else:
          print(f"{len(unmatched_results)} unmatched results found")

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(unmatched_results)
        print(df)

        return unmatched_results


class CountDuplicates:
    """
    Initializes the CountDuplicates with a list of dictionaries.

    Args:
        list_of_dicts (List[Dict]): A list of dictionaries.
        key (str): The key whose values will be checked for duplicates.
    """
    def __init__(self, list_of_dicts, key):
        self.list_of_dicts = list_of_dicts
        self.key = key

    def count_duplicate_strings(self):
        """
        Counts duplicate strings in a list of dictionaries and filters out those with only 1 occurrence.

        Returns:
            dict: A dictionary with strings as keys and their counts as values (only if count > 1).
        """
        # Collect all strings from the specified key in each dictionary
        strings = [d[self.key] for d in self.list_of_dicts if self.key in d]

        # Use Counter to count occurrences and filter out those with count 1
        counted_strings = Counter(strings)

        # Filter out strings with a count of 1
        print(f"Total sum of counts: {counted_strings.total()}")
        return counted_strings

    def filter_dict(self):
        """
        Filters out strings with a count of 1.

        Returns:
            dict: A dictionary with strings as keys and their counts as values (only if count > 1).
        """
        # Use Counter to count occurrences and filter out those with count 1
        counted_strings = self.count_duplicate_strings()

        filtered_dict = {string: count for string, count in counted_strings.items() if count > 1}

        # Print total sum of filtered counts
        print(f"Total sum of filtered counts: {sum(filtered_dict.values())}")

        # Filter out strings with a count of 1
        return filtered_dict

    def top_duplicates(self, top_n = None):
        """
        Return a list of the n most common elements and their counts from the
        most common to the least. If n is omitted or None, most_common() returns
        all elements in the counter. Elements with equal counts will be ordered
        in the order first encountered:

        Args:
            top_n (int): The number of most common elements to return.

        Returns:
            list: A list of the n most common elements and their counts.
        """

        counted_strings = self.count_duplicate_strings()
        return counted_strings.most_common(top_n)
class LLMTextDiversityAnalyzer:
    def __init__(self, thematic_analysis):
        """
        Initializes the class and sets the thematic analysis instance.

        Args:
            thematic_analysis: Instance of ThematicAnalysis to generate themes and codes.
        """
        self.thematic_analysis = thematic_analysis  # ThematicAnalysis instance

    def run_thematic_analysis(self, runs=10, filename: Optional[str] = None):
        """
        Executes the thematic analysis by calling the zs_codes method.

        Args:
            runs (int): Number of times to run the thematic analysis.
            filename: Optional filename to save the thematic analysis result.

        Returns:
            The result of the thematic analysis.
        """
        codes = []
        for i in range(runs):
            try:
                codes.append(self.thematic_analysis.generate_codes())
                print(f"Thematic analysis {i+1} successfully run.")
            except Exception as e:
                print(f"Error running thematic analysis {i}: {e}")
                raise
        # Save results to file
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(codes, f, indent=4)
                print(f"Results successfully saved to {filename}")
        self.zs_code_results = codes
        return codes

    def set_code_data(self):
        """
        Extracts relevant theme data (theme, subthemes, codes) from the zs_control_gemini output.
        """
        if not hasattr(self, 'zs_code_results'):
            raise ValueError("Thematic analysis has not been run yet. Run run_thematic_analysis() first.")

        # t5p5[0][0]['code']
        all_runs = []
        for result in self.zs_code_results:
          all_codes = ""
          # Access the elements within the nested structure using their appropriate index
          for i in result:
            all_codes += i['code'] + " "
          all_runs.append(all_codes)
        self.all_runs = all_runs
        return all_runs

    def count_tokens(self):
        """Counts the number of tokens in a text."""
        all_tokens = []
        num_of_tokens = []
        for i, run in enumerate(self.all_runs):
            tokens = nltk.word_tokenize(run)
            all_tokens.append(tokens)
            num_of_tokens.append(len(tokens))
            print(f"Run {i + 1} token count: {len(tokens)}")
        self.all_tokens = all_tokens
        self.num_of_tokens = num_of_tokens
        return all_tokens, num_of_tokens

    def count_unique_ngrams(self, n=2):
        """Counts the unique n-grams (bi-grams, tri-grams, etc.) in a text."""
        ngrams = []
        unique_ngram_count = []
        for i, tokens in enumerate(self.all_tokens):
            n_grams = list(nltk_ngrams(tokens, n))
            ngrams.append(n_grams)
            unique_ngrams = set(n_grams)
            unique_ngram_count.append(len(unique_ngrams))
            print(f"Unique {n}-grams in run {i + 1}: {len(unique_ngrams)}")
        if n == 2:
            self.bigrams = ngrams
            self.unique_bigram_count = unique_ngram_count
        elif n == 3:
            self.trigrams = ngrams
            self.unique_trigram_count = unique_ngram_count
        else:
            self.ngrams = ngrams
            self.unique_ngram_count = unique_ngram_count
        return ngrams, unique_ngram_count

    def display_results(self):
        """Displays diversity metrics."""
        df = pd.DataFrame({
            "Tokens": self.all_tokens,
            "Token Count": self.num_of_tokens,
            "Bigrams": self.bigrams,
            "Unique Bigrams": self.unique_bigram_count,
            "Trigrams": self.trigrams,
            "Unique Trigrams": self.unique_trigram_count
        })

        print(df.describe())
        return df


class QA_CoupleGenerator:
    def __init__(self, repo_id: str, n_generations: int = 10, timeout: int = 120):
        """
        Initializes the QA Couple Generator with the given model repository ID and generation settings.

        Args:
            repo_id (str): The model repository ID on HuggingFace.
            n_generations (int): Number of QA couples to generate.
            timeout (int): Timeout for the inference client in seconds.
        """
        self.repo_id = repo_id
        self.n_generations = n_generations
        self.llm_client = InferenceClient(model=repo_id, timeout=timeout)
        notebook_login()

    def call_llm(self, prompt: str) -> Tuple[str, str]:
        """
        Calls the LLM with the provided prompt and parses the generated factoid question and answer.

        Args:
            prompt (str): The formatted input prompt for the model.

        Returns:
            Tuple[str, str]: The generated factoid question and answer.
        """
        response = self.llm_client.post(
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1000},
                "task": "text-generation",
            },
        )
        output_QA_couple = json.loads(response.decode())[0]["generated_text"]
        question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
        answer = output_QA_couple.split("Answer: ")[-1]
        return question.strip(), answer.strip()

    def generate_QA_couples(self, contexts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generates QA couples using the provided list of contexts.

        Args:
            contexts (List[str]): List of context strings to use for QA generation.

        Returns:
            Tuple[List[str], List[str]]: Lists of generated questions and ground truths (answers).
        """
        QA_generation_prompt = """
        Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

        Provide your answer as follows:

        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)

        Now here is the context.

        Context: {context}\n
        Output:::"""

        questions = []
        ground_truths = []

        print(f"Generating {self.n_generations} QA couples...")

        for sampled_context in tqdm(random.sample(contexts, self.n_generations)):
            try:
                question, answer = self.call_llm(QA_generation_prompt.format(context=sampled_context))
                questions.append(question)
                ground_truths.append(answer)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

        return questions, ground_truths

    def save_dataset(self, questions: List[str], ground_truths: List[str], filename: Optional[str] = None) -> dict:
        """
        Prepares a dataset dictionary with empty answers and contexts, ready for later use.

        Args:
            questions (List[str]): List of generated factoid questions.
            ground_truths (List[str]): List of corresponding ground truths (answers).
            filename (Optional[str]): Optional filename to save the dataset.
        Returns:
            dict: A dictionary formatted for dataset use with empty answers and contexts.
        """
        data = {
            "question": questions,
            "answer": [""] * len(questions),  # Empty answers for later inference
            "contexts": [""] * len(questions),  # Empty contexts for later retrieval
            "reference": ground_truths
        }

        # Save results to file
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Data successfully saved to {filename}")
            else:
                print("Invalid file format. Please use .json")

        return data


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


class RAGAsEvaluation:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = self._build_prompt_template()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def _build_rag_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run_inference(self, data_dict: dict) -> dict:
        """
        Runs inference on a dataset where 'answer' and 'contexts' are initially blank.

        Args:
            data_dict (dict): A dictionary with the fields 'question', 'answer', 'contexts', and 'reference'.
                              'answer' and 'contexts' should be blank and will be populated during inference.

        Returns:
            dict: The updated dictionary with answers and contexts filled in.
        """
        rag_chain = self._build_rag_chain()

        # Extracting questions from the dictionary
        questions = data_dict.get("question", [])
        answers = []
        contexts = []

        # Running inference for each question
        for query in questions:
            answers.append(rag_chain.invoke(query))
            contexts.append([doc.page_content for doc in self.retriever.get_relevant_documents(query)])

        # Update the dictionary with the new answers and contexts
        data_dict["answer"] = answers
        data_dict["contexts"] = contexts

        return data_dict

    def evaluate(self, data_dict: dict) -> 'pd.DataFrame':
        dataset = Dataset.from_dict(data_dict)
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        return result.to_pandas()

    def summarize_results(self, results_df: pd.DataFrame, box_title: str = "RAGAs Evaluation Metric Distribution"):
        """
        Summarizes and visualizes the evaluation results with customizable graph titles.

        Args:
            results_df (pd.DataFrame): The DataFrame containing evaluation results.
            bar_title (str): Title for the bar chart. Defaults to "RAGAs Evaluation Metrics".
            box_title (str): Title for the box plot. Defaults to "RAGAs Evaluation Metric Distribution".
        """
        # Summary statistics (mean, median, etc.)
        summary = results_df.describe()

        print("Summary of Results:")
        print(summary)

        # Create visualizations with custom titles
        self._visualize_results(results_df, box_title)

    def _visualize_results(self, results_df: pd.DataFrame, box_title: str):
        """
        Generates bar charts and box plots with custom titles for each evaluation metric.

        Args:
            results_df (pd.DataFrame): The DataFrame containing evaluation results.
            bar_title (str): Title for the bar chart.
            box_title (str): Title for the box plot.
        """
        # sns.set(style="whitegrid")

        # Generate bar plots for each metric
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        results_long = results_df.melt(value_vars=metrics, var_name="Metric", value_name="Score")

        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Metric", y="Score", data=results_long, palette="coolwarm")
        plt.title(box_title)
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()