# prompting_methods.py

from langchain.prompts import PromptTemplate


class ZeroShotPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs):
        zs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions, \
        generate themes with definitions and supporting quotes from the text: {data}."""
        zs_prompt = PromptTemplate.from_template(zs_template)
        chain = zs_prompt | self.llm
        results = chain.invoke({"rqs": rqs, "data": data})
        return results


class FewShotPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, examples):
        """Generate response using a few-shot prompt."""
        fs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions and the examples provided: {examples}, \
        generate themes with definitions and supporting quotes from the text: {data}."""
        fs_prompt = PromptTemplate.from_template(fs_template)
        chain = fs_prompt | self.llm
        return chain.invoke({"rqs": rqs, "data": data, "examples": examples})


class ChainOfThoughtPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, chain_template):
        # Implement cot prompting method here
        pass

