#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import langchain
import langchain_core
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class ModelManager:
    def __init__(self, model_choice='gemini-1.5-flash', text=None):
        load_dotenv()
        self.model_choice = model_choice
        self.text = text
        self.llm = self._initialize_model(model_choice)

    def _initialize_model(self, model_choice):
        if model_choice == 'gemini-1.5-flash':
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            return ChatGoogleGenerativeAI(model=model_choice, temperature=0.7, google_api_key=gemini_api_key, top_p=0.3)
        elif model_choice == 'gpt-4o-2024-05-13':
            openai_api_key = os.getenv('OPENAI_API_KEY')
            return ChatOpenAI(model=model_choice, temperature=0.7, api_key=openai_api_key, top_p=0.3)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")


if __name__ == "__main__":
    model_choice = 'gemini-1.5-flash' or 'gpt-4o-2024-05-13'
    model = Model(model_choice=model_choice, text="Hello, world!")
    print(model.llm)  # This will print the LLM instance based on the choice



