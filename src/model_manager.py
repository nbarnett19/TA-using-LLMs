# model_manager.py

import getpass
import os
from dotenv import load_dotenv
import langchain
import langchain_core
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class ModelManager:
    """
    Manages the initialization and configuration of different language models.

    Attributes:
        model_choice (str): The choice of model to initialize.
        text (str): Optional text to initialize with the model.
        llm: The initialized language model.
    """

    def __init__(self, model_choice='gemini-1.5-flash', text=None):
        """
        Initializes the ModelManager with the given model choice and text.

        Args:
            model_choice (str): The choice of model to initialize.
            text (str): Optional text to initialize with the model.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Ensure API keys are available
        self._ensure_api_keys()

        self.model_choice = model_choice
        self.text = text
        self.llm = self._initialize_model(model_choice)

    def _ensure_api_keys(self):
        """
        Prompts the user to enter API keys if they are not set in environment variables.
        """
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")

        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OpenAI API Key: ")

    def _initialize_model(self, model_choice):
        """
        Initializes the appropriate language model based on the model choice.

        Args:
            model_choice (str): The choice of model to initialize.

        Returns:
            An instance of the chosen language model.

        Raises:
            ValueError: If an unknown model choice is provided.
        """
        if model_choice.startswith('gemini'):
            gemini_api_key = os.getenv('GOOGLE_API_KEY')
            if not gemini_api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set in environment variables")
            return ChatGoogleGenerativeAI(model=model_choice, temperature=0.7, google_api_key=gemini_api_key, top_p=0.3)
        elif model_choice == 'gpt-4o-2024-05-13':
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
            return ChatOpenAI(model=model_choice, temperature=0.7, api_key=openai_api_key, top_p=0.3)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")


# Example Usage
if __name__ == "__main__":
    model_manager = ModelManager(model_choice='gemini-1.5-flash')
    print(model_manager.llm)  # This will print the initialized model instance




