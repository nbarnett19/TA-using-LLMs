# main.py

import langchain
import langchain_core
import langchain_community
from src import ModelManager, ZeroShotPrompt, TranscriptLoader

# Initialize the ModelManager
model_manager = ModelManager(model_choice='gemini-1.5-flash')

# Load the transcript data from the PDF
transcript_loader = TranscriptLoader("data/Alle_Transkripte_EN.pdf")
data = transcript_loader.load_text_from_pdf()

# Define the research questions
rqs = """Explore and describe experiences of internal medicine doctors after wearing a
glucose sensor with focus on two research questions:
1. How can self-tracking with a glucose sensor influence residents’ understanding of glucose metabolism?
2. How can self-tracking with a glucose sensor improve residents’ awareness, appreciation, and
understanding of patients with diabetes?"""

# Initialize the ZeroShotPrompt with the chosen language model
zs_prompt = ZeroShotPrompt(llm=model_manager.llm)

# Generate a response
try:
    response = zs_prompt.generate_response(data=data, rqs=rqs)
    print(response)
except Exception as e:
    print(f"Failed to generate response: {e}")

