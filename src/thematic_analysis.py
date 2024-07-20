# main.py

import os
from dotenv import load_dotenv
import langchain
import langchain_core
import langchain_community
from transcript_loader import TranscriptLoader
from model_manager import ModelManager
from prompting_methods import ZeroShotPrompt, FewShotPrompt, ChainOfThoughtPrompt

# gemini_api_key = os.getenv('GEMINI_API_KEY')
# ChatGoogleGenerativeAI(model=model_choice, temperature=0.7, google_api_key=gemini_api_key, top_p=0.3)

def main():
    # Define file path and research questions
    file_path = "./Data/Alle_Transkripte_EN.pdf"
    rqs = """Explore and describe experiences of internal medicine doctors after wearing a
    glucose sensor with focus on two research questions:
    1. How can self-tracking with a glucose sensor influence residents’ understanding of glucose metabolism?
    2. How can self-tracking with a glucose sensor improve residents’ awareness, appreciation, and
    understanding of patients with diabetes?"""

    # Choose model and method for testing
    model_choice = 'gemini-1.5-flash'  # Change if needed
    method = 'zero-shot'  # Choose from 'zero-shot', 'few-shot', or 'chain-of-thought'

    # Initialize ModelManager
    connect = ModelManager(model_choice=model_choice)

    # Load data
    pdf = TranscriptLoader(file_path)
    data = pdf.load_text_from_pdf()

    # Perform analysis
    if method == 'zero-shot':
        prompt = ZeroShotPrompt(connect.llm)
        results = prompt.generate_response(data, rqs)
    elif method == 'few-shot':
        examples = "Example 1: Theme: X, Definition: ... Example 2: Theme: Y, Definition: ..."
        prompt = FewShotPrompt(connect.llm)
        results = prompt.generate_response(data, rqs, examples)
    elif method == 'chain-of-thought':
        chain_template = """You are a researcher. Based on the research questions: {rqs}
        and data: {data}, generate themes with definitions and supporting quotes."""
        prompt = ChainOfThoughtPrompt(connect.llm)
        results = prompt.generate_response(data, rqs, chain_template)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Print results
    if results:
        print(f"Results using {model_choice} with {method} method:")
        print(results)
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()
