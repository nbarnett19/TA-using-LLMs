# main.py

import os
from dotenv import load_dotenv
from src import ModelManager, FocusGroup, CodeExcerpt, Themes, ZSControl, FolderLoader, ScannedPDFLoader, \
    ThematicAnalysis, GenerateCodes, GenerateThemes, QuoteMatcher, CountDuplicates, LLMTextDiversityAnalyzer, \
    QA_CoupleGenerator, ChromaVectorStoreManager, RAGAsEvaluation

# Initialize the ModelManager
model_manager = ModelManager(model_choice='gemini-1.5-pro', temperature=0.5, top_p=0.5)

# Load the transcript data from the PDF
transcript_loader = TranscriptLoader("data/Alle_Transkripte_EN.pdf")
text = transcript_loader.load_text_from_pdf()

# Split the text into chunks
chunks = transcript_loader.split_text_into_chunks(chunk_size=10000, chunk_overlap=1000)
print("Number of chunks:", len(chunks))


# Define the research questions
rqs = """Explore and describe experiences of internal medicine doctors after wearing a
glucose sensor with focus on two research questions:
1. How can self-tracking with a glucose sensor influence residents’ understanding of glucose metabolism?
2. How can self-tracking with a glucose sensor improve residents’ awareness, appreciation, and
understanding of patients with diabetes?"""

# Initialize the thematic analysis with the chosen language model
prompt = ThematicAnalysis(llm=model_manager.llm, chunks=chunks, rqs=rqs)

# Generate data summary
summary = prompt.generate_summary()

# Generate Codes
df = prompt.generate_codes(filename="test_results/generate_code_test2.json")

# Zero-shot prompt themes
zs_results = prompt.zs_prompt(df, filename="test_results/zs_themes_test2.json")

# Format themes into hierarchical dataframe
HierarchicalDataFrame(zs_results).get_hierarchical_df(filename="test_results/zs_themes_test2", file_format="hdf")
HierarchicalDataFrame(zs_results).get_hierarchical_df(filename="test_results/zs_themes_test2", file_format="csv")