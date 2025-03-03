# main.py
import argparse
import os
from dotenv import load_dotenv
from src import ModelManager, FocusGroup, CodeExcerpt, Themes, ZSControl, FolderLoader, ScannedPDFLoader, \
    ThematicAnalysis, GenerateCodes, GenerateThemes, QuoteMatcher, CountDuplicates, LLMTextDiversityAnalyzer, \
    QA_CoupleGenerator, ChromaVectorStoreManager, RAGAsEvaluation

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

    print(f"Analysis complete! Themes saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run reflexive thematic analysis using LLMs.")
    parser.add_argument("--data", type=str, required=True, help="Path to the folder containing transcript files.")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro",
                        help="LLM model to use (default: gemini-1.5-pro)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for LLM responses (default: 0.5)")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p sampling value (default: 0.5)")
    parser.add_argument("--output", type=str, default="themes.json",
                        help="Output file for generated themes (default: themes.json)")

    args = parser.parse_args()
    run_analysis(args.data, args.model, args.temperature, args.top_p, args.output)


if __name__ == "__main__":
    main()
