# qa_generation.py

from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
from huggingface_hub import InferenceClient, notebook_login
import random

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