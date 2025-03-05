# Conducting Qualitative Thematic Analysis with Large Language Models and RAG Implementation

## Overview
This repository contains the code and methodology for conducting Reflexive Thematic Analysis (RTA) using Large Language Models (LLMs) with and without Retrieval-Augmented Generation (RAG). The research explores how LLMs, specifically **GPT-4o** and **Gemini 1.5 Pro**, can perform inductive thematic analysis on focus group transcripts.

The project evaluates **zero-shot**, **few-shot**, and **chain-of-thought (CoT)** prompting techniques to determine the most effective approach for qualitative analysis. Additionally, it examines whether **RAG** improves the quality of LLM-generated themes.

## Research Questions
1. How should LLM prompts be formulated to achieve the best results for thematic analysis?
2. How does Retrieval-Augmented Generation (RAG) influence LLM-generated thematic outputs?
3. How do different LLMs (GPT-4o vs. Gemini 1.5 Pro) compare in thematic analysis performance?

## Dataset
The focus group data consists of transcripts from **medical doctors** discussing their experiences wearing glucose monitoring sensors. The dataset includes **38,000 words**, originally in German and translated into English using DeepL Pro.

## Methodology
The project follows a **three-phase methodology**:
1. **Prompt Engineering** – Developing effective thematic analysis prompts (zero-shot, few-shot, and CoT).
2. **RAG Implementation** – Fine-tuning RAG architecture for enhanced qualitative analysis.
3. **Model Comparison** – Comparing outputs between **GPT-4o** and **Gemini 1.5 Pro**.

### Technologies Used
- **Python** (Executed in Google Colab Pro)
- **LangChain** for LLM workflow integration
- **OpenAI & Google API** for LLM access
- **Chroma** as a vector database
- **Tesseract OCR & Poppler-utils** for document parsing
- **Pandas, NumPy** for data processing

## Key Findings
- **CoT prompting** produces the most analytically coherent themes.
- **RAG** does not consistently improve results and may introduce noise.
- **GPT-4o and Gemini 1.5 Pro** exhibit strong thematic alignment, with minor variations in interpretation.
- **Embedding Models**: OpenAI's `text-embedding-3-large` and Google's `text-embedding-004` were tested for optimal vector representation.

## Installation
### System Dependencies
Before running the code, install the required system dependencies:
```sh
sudo apt update && sudo apt install -y poppler-utils tesseract-ocr libtesseract-dev libleptonica-dev
```

### Python Dependencies
Install the necessary Python packages:
```sh
pip install -r requirements.txt
```

## Usage
1. Install package from PyPi:
```
pip install TA_using_LLMs
```
OR

Clone the repository:
   ```sh
   git clone https://github.com/nbarnett19/Thematic_Analysis_using_LLMs.git
   cd Thematic_Analysis_using_LLMs
   ```
2. Install dependencies (see above).
3. Link to Colab Demo: https://colab.research.google.com/drive/19MrRwsY0dn3rtzGQUKtI1Ubyb0Swz0Rw?usp=sharing 

Repository Structure
```
├── RAG_files/                   # Data files used for retrieval-augmented generation
├── ScannedPDFs_for_RAG/         # Additional scanned documents for retrieval
├── TA_using_LLMs/               # Core thematic analysis module
├── data/                        # Uploaded data files for analysis
├── dist/                        # Distribution package files
├── .github/                     # GitHub workflow and configuration files
├── .gitignore                   # Ignore rules for Git
├── LICENSE                      # Project license
├── README.md                    # This README file
├── TA_using_LLMs_DEMO.ipynb      # Jupyter Notebook for demonstration
├── pyproject.toml                # Project metadata and build system requirements
├── requirements.txt              # List of dependencies
├── setup.py                      # Installation script
```

## Contributors
- **Natalie A. Barnett** (Author)
- **Lucerne University of Applied Sciences and Arts**
- **Lecturers: Rabea Krings & Diego Antognini**

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Citation
If you use this research, please cite:
```
Barnett, N.A. (2024). Conducting Qualitative Thematic Analysis with Large Language Models and RAG Implementation. MSc Thesis, Lucerne University of Applied Sciences and Arts.
```

---
