#setup.py
from setuptools import setup, find_packages
import subprocess
import sys

# Function to check if a system package is installed
def check_system_dependency(command, package_name):
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        sys.stderr.write(f"Warning: {package_name} is not installed. Please install it using:\n")
        sys.stderr.write(f"  sudo apt install {package_name}\n")
        sys.exit(1)

# Check for required system packages
check_system_dependency(["which", "tesseract"], "tesseract-ocr")
check_system_dependency(["which", "pdftotext"], "poppler-utils")

setup(
    name="TA_using_LLMs",
    version="0.0.4",
    authors="Natalie Barnett",
    author_email="nataliebarnett.ch@gmail.com",
    description="An application that performs qualitative thematic analysis using LLMs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "chromadb==0.6.3",
        "datasets==3.3.2",
        "fuzzywuzzy==0.18.0",
        "huggingface-hub==0.28.1",
        "langchain==0.3.19",
        "langchain-chroma==0.2.2",
        "langchain-community==0.3.18",
        "langchain-core==0.3.37",
        "langchain-experimental==0.3.4",
        "langchain-google-genai==2.0.9",
        "langchain-openai==0.3.6",
        "langchain-text-splitters==0.3.6",
        "Levenshtein==0.26.1",
        "matplotlib",
        "multiprocess==0.70.16",
        "nltk==3.9.1",
        "numpy==1.26.4",
        "openai==1.61.1",
        "opencv-python==4.11.0.86",
        "pandas==2.2.2",
        "pdf2image==1.17.0",
        "pillow==11.1.0",
        "pydantic==2.10.6",
        "pydantic-settings==2.8.0",
        "pydantic_core==2.27.2",
        "pypdf==5.3.0",
        "pytesseract==0.3.13",
        "python-dotenv==1.0.1",
        "python-Levenshtein==0.26.1",
        "python-utils==3.9.1",
        "ragas==0.2.13",
        "scikit-learn==1.6.1",
        "seaborn==0.13.2",
        "sentence-transformers==3.4.1",
        "tensorflow==2.18.0",
        "tqdm==4.67.1",
        "transformers==4.48.3",
    ],
    entrypoints={"console_scripts": ["TA_using_LLMs = TA_using_LLMs.main:main"]},
    homepage="https://github.com/nbarnett19/Thematic_Analysis_using_LLMs",
)