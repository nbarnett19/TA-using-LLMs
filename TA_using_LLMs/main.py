# TA_using_LLMs\main.py
import argparse
import pandas as pd
from dotenv import load_dotenv
from src.TA_using_LLMs.logic import ModelManager, FolderLoader, ThematicAnalysis

# Load environment variables if needed
load_dotenv()


def run_analysis(data_path, model_choice, temperature, top_p, rqs, filename):
    """
    Runs reflexive thematic analysis on the provided data using an LLM.

    :param data_path: Path to the directory containing focus group transcripts
    :param model_choice: Name of the LLM to use (e.g., 'gemini-1.5-pro')
    :param temperature: Sampling temperature for LLM responses
    :param top_p: Top-p nucleus sampling for LLM responses
    :param rqs: Research questions of the thematic analysis
    :param filename: Output json filename to save themes
    """

    print("Initializing ModelManager...")
    model_manager = ModelManager(model_choice=model_choice, temperature=temperature, top_p=top_p)

    print(f"Loading data from {data_path}...")
    loader = FolderLoader(data_path)
    docs = loader.load_txt()

    print("Splitting text into chunks...")
    chunks = loader.split_text(docs)
    print(f"Number of chunks: {len(chunks)}")

    print("Performing thematic analysis...")
    prompt = ThematicAnalysis(llm=model_manager.llm, docs=docs, chunks=chunks, rqs=rqs)
    results = prompt.zs_control_gemini(filename=filename)
    pd.json_normalize(results)

    print(f"Analysis complete! Themes saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run reflexive thematic analysis using LLMs.")
    parser.add_argument("--data", type=str, default="data", help="Path to the folder containing transcript files.")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro",
                        help="LLM model to use (default: gemini-1.5-pro)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for LLM responses (default: 0.5)")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p sampling value (default: 0.5)")
    parser.add_argument("--rqs", type=str, default="Explore and describe experiences of internal medicine doctors after "
                                                   "wearing a glucose sensor with focus on two research questions: 1. How "
                                                   "can self-tracking with a glucose sensor influence residents’ "
                                                   "understanding of glucose metabolism? 2. How can self-tracking with a "
                                                   "glucose sensor improve residents’ awareness, appreciation, and "
                                                   "understanding of patients with diabetes?",
                        help="Research questions for the thematic analysis (default: None)")
    parser.add_argument("--filename", type=str, default="themes.json",
                        help="Output file for generated themes (default: themes.json)")

    args = parser.parse_args()
    run_analysis(args.data, args.model, args.temperature, args.top_p, args.rqs, args.filename)


if __name__ == "__main__":
    main()
