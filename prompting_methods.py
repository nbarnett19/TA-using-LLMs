# prompting_methods.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ZeroShotPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs):
        zs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions, \
        generate themes with definitions and supporting quotes from the text: {data}."""
        zs_prompt = PromptTemplate.from_template(zs_template)
        chain = LLMChain(llm=self.llm, prompt=zs_prompt)
        return chain.run({"rqs": rqs, "data": data})


class FewShotPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, examples):
        # Implement few-shot prompting method here
        pass


class ChainOfThoughtPrompt:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, data, rqs, chain_template):
        chain_prompt = PromptTemplate.from_template(chain_template)
        chain = LLMChain(llm=self.llm, prompt=chain_prompt)
        return chain.run({"rqs": rqs, "data": data})
