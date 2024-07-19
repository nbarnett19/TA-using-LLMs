# main.py

from transcript_loader import TranscriptLoader
from model_manager import ModelManager
from thematic_analysis import ThematicAnalysis


def load_transcript(file_path: str) -> str:
    """Load the transcript data from the PDF file."""
    transcript_loader = TranscriptLoader(file_path)
    return transcript_loader.load_text_from_pdf()


def perform_analysis(data: str, rqs: str, model_choice: str, method: str):
    """Perform thematic analysis using the specified model and method."""
    model_manager = ModelManager(model_choice=model_choice)
    thematic_analysis = ThematicAnalysis(model_manager)

    # Depending on the method, you might pass examples or chain templates
    if method == 'few-shot':
        examples = "Example 1: Theme: X, Definition: ... Example 2: Theme: Y, Definition: ..."
        results = thematic_analysis.perform_analysis(data, rqs, method='few-shot', examples=examples)
    elif method == 'chain-of-thought':
        chain_template = """You are a researcher. Based on the research questions: {rqs} and data: {data}, 
        generate themes with definitions and supporting quotes."""
        results = thematic_analysis.perform_analysis(data, rqs, method='chain-of-thought', chain_template=chain_template)
    else:
        results = thematic_analysis.perform_analysis(data, rqs, method='zero-shot')

    return results


def main():
    # Define file path and research questions
    file_path = "./Data/Alle_Transkripte_EN.pdf"
    rqs = """Explore and describe experiences of internal medicine doctors after wearing a 
    glucose sensor with focus on two research questions:
    1. How can self-tracking with a glucose sensor influence residents’ understanding of glucose metabolism?
    2. How can self-tracking with a glucose sensor improve residents’ awareness, appreciation, and 
    understanding of patients with diabetes?"""

    # Choose model and method for testing
    model_choice = 'gpt-4o-2024-05-13'  # Change to 'gemini-1.5-flash' if needed
    method = 'zero-shot'  # Choose from 'zero-shot', 'few-shot', or 'chain-of-thought'

    # Load data
    data = load_transcript(file_path)

    # Perform analysis
    results = perform_analysis(data, rqs, model_choice, method)

    # Print results
    print(f"Results using {model_choice} with {method} method:")
    print(results)


if __name__ == "__main__":
    main()
