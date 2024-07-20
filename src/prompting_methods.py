# prompting_methods.py

from langchain.prompts import PromptTemplate


class ZeroShotPrompt:
    """
    Generates responses using a zero-shot prompt technique.

    Attributes:
        llm: The language model used to generate responses.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs):
        """
        Generates themes with definitions and supporting quotes from the text.

        Args:
            data (str): The text data to analyze.
            rqs (str): The research questions to answer.

        Returns:
            The generated response from the language model.
        """
        zs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions, \
        generate themes with definitions and supporting quotes from the text: {data}."""
        zs_prompt = PromptTemplate.from_template(zs_template)
        chain = zs_prompt | self.llm

        try:
            results = chain.invoke({"rqs": rqs, "data": data})
            print("Results received from the model:", results)
            return results
        except Exception as e:
            print(f"Error occurred: {e}")
            raise


class FewShotPrompt:
    """
    Generates responses using a few-shot prompt technique.

    Attributes:
        llm: The language model used to generate responses.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, examples):
        """
        Generates themes with definitions and supporting quotes from the text.

        Args:
            data (str): The text data to analyze.
            rqs (str): The research questions to answer.
            examples (str): Example responses to guide the language model.

        Returns:
            The generated response from the language model.
        """
        fs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions and the examples provided: {examples}, \
        generate themes with definitions and supporting quotes from the text: {data}."""
        fs_prompt = PromptTemplate.from_template(fs_template)
        chain = fs_prompt | self.llm
        return chain.invoke({"rqs": rqs, "data": data, "examples": examples})


class ChainOfThoughtPrompt:
    """
    Placeholder class for Chain of Thought (CoT) prompting method.

    Attributes:
        llm: The language model used to generate responses.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, chain_template):
        """
        Placeholder method for CoT prompting.

        Args:
            data (str): The text data to analyze.
            rqs (str): The research questions to answer.
            chain_template (str): Template for CoT prompting.

        Returns:
            The generated response from the language model.
        """
        pass

