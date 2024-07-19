# thematic_analysis.py

from prompting_methods import ZeroShotPrompt, FewShotPrompt, ChainOfThoughtPrompt
from langchain.prompts import PromptTemplate


class ThematicAnalysis:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def perform_analysis(self, data: str, rqs: str, method: str, examples: str = None,
                         chain_template: str = None) -> str:
        """Perform thematic analysis using the specified method."""
        if method == 'zero-shot':
            # Directly generate response without examples
            prompt = PromptTemplate.from_template(
                "You are a researcher. Based on the research questions: {rqs} and data: {data}, generate themes."
            )
            result = self.model_manager.llm.generate_response(prompt.generate_prompt(data, rqs))
        elif method == 'few-shot':
            # Use examples if provided
            if examples:
                prompt = PromptTemplate.from_template(
                    "You are a researcher. Based on the research questions: {rqs} and data: {data}, generate themes using these examples: {examples}."
                )
                result = self.model_manager.llm.generate_response(prompt.generate_prompt(data, rqs, examples=examples))
            else:
                raise ValueError("Examples must be provided for few-shot method.")
        elif method == 'chain-of-thought':
            # Use chain template if provided
            if chain_template:
                prompt = PromptTemplate.from_template(chain_template)
                result = self.model_manager.llm.generate_response(prompt.generate_prompt(data, rqs))
            else:
                raise ValueError("Chain template must be provided for chain-of-thought method.")
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

