# thematic_analysis.py

import pandas as pd
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json


class ThematicAnalysis:
    """
    Generates themes with definitions and supporting quotes from the text.

    Attributes:
        llm: The language model used to generate responses.
        docs (list):The full documents to analyze.
        chunks (list): The text chunks to analyze.
        rqs (str): The research questions to answer.
    """
    def __init__(self, llm, docs, chunks, rqs):
        self.llm = llm
        self.docs = docs
        self.chunks = chunks
        self.rqs = rqs

    def generate_summary(self):
        """
        Generates summary from the text based on research questions.

        Returns:
            The generated response from the language model for the combined text.
        """
        template = """You are a qualitative researcher. The aim of your study is
        to answer the following research questions: {rqs}
        Based on your research questions, generate a short summary from the text: {text}."""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm

        summaries = []
        for chunk in self.docs:
            try:
                # Generate summary for each chunk
                summary = chain.invoke({"rqs": self.rqs, "text": chunk.page_content})
                summaries.append(summary.content)
            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        # Combine all chunk summaries into a single summary
        combined_summary_template = """You are a qualitative researcher.
        The aim of your study is to answer the following research questions: {rqs}
        Based on your research questions and the following summaries,
        generate an overall summary: {summaries}."""
        combined_summary_prompt = PromptTemplate.from_template(combined_summary_template)
        combined_chain = combined_summary_prompt | self.llm

        try:
            final_summary = combined_chain.invoke({"rqs": self.rqs, "summaries": "\n\n".join(summaries)})
            print(final_summary.usage_metadata)
            print("Final summary:", final_summary.content)
            return final_summary.content
        except Exception as e:
            print(f"Error occurred while generating final summary: {e}")
            raise

    def zs_control_gemini(self, filename=None) -> Any:
        """
        Generates themes with definitions, subthemes with definitions, codes and excerpts.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """

        prompt_template = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Based on the research questions provided, you must identify a maximum of 6 distinct themes.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The transcripts: {text}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["rqs", "format_instructions", "text"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        # Initialize an empty str
        all_text = ""

        # Iterate through each focus group in the data
        for item in self.docs:
            # Iterate through the content of the focus group
            for text in item.page_content:
              # Append each text to the text_chunk string
              all_text += text

        try:
          results = chain.invoke({
              "rqs": self.rqs,
              "text": all_text
              })

          print(prompt.template.format(
                rqs=self.rqs,
                text=all_text,
                format_instructions=format_instructions
            ))

          if filename:
              if filename.endswith('.json'):
                  with open(filename, 'w') as f:
                      json.dump(results, f, indent=4)
                      print(f"Results successfully saved to {filename}")
              elif filename.endswith('.csv'):
                  df = pd.DataFrame(results)
                  df.to_csv(filename, index=False)
                  print(f"Results successfully saved to {filename}")
              else:
                  print("Invalid file format. Please use .json or .csv.")

          return results

        except Exception as e:
            print(f"Error occurred while processing: {e}")
            raise

    def zs_control_gpt(self, filename=None) -> Any:
        """
        Generates themes with definitions, subthemes with definitions, codes, and excerpts.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            A single JSON object containing all themes, sub-themes, and codes across all chunks.
        """

        prompt_template = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Based on the research questions provided, you must identify a maximum of 6 distinct themes.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The transcripts: {text}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["rqs", "format_instructions", "text"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_themes = []

        for data in self.docs:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing file: {source_file}")

                # Extract text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }

                # Generate themes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a list, use it directly
                if isinstance(response, list):
                    themes = response
                else:
                    # If the response is a single dictionary, wrap it in a list
                    if isinstance(response, dict):
                        themes = [response]
                    else:
                        raise ValueError("Unexpected format: response must be a list or dictionary.")

                # Ensure themes are in the expected format
                for theme in themes:
                    if not isinstance(theme, dict) or not all(key in theme for key in ["theme", "theme_definition", "subthemes", "subtheme_definitions", "codes", "supporting_quotes", "speaker"]):
                        raise ValueError("Invalid theme format detected.")

                    # Add focus group information
                    theme['source file'] = source_file

                # Flatten the results
                all_themes.extend(themes)

            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        # Second prompt for refining and filtering themes
        prompt_template2 = """You are a qualitative researcher doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Provided are a list previously
        identified themes from the data. Review the given list of themes and combine or
        filter them to identify a maximum of 6 distinct themes that address the research questions.
        Each theme should include:
        1. A theme definition
        2. A sub-theme if needed
        3. Each sub-theme should have a definition
        4. Supporting codes for each sub-theme
        5. Each code should be supported with a word for word excerpt from the
        transcript and excerpt speaker from the text.
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.
        Please ensure that the themes are clearly distinct and cover various aspects of the data.
        Follow this format: {format_instructions}.
        Research questions: {rqs}
        The list of themes: {themes}"""

        parser = JsonOutputParser(pydantic_object=ZSControl)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template2,
            input_variables=["rqs", "format_instructions", "themes"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        # Prepare input dictionary
        input_data = {
            "rqs": self.rqs,
            "themes": all_themes
        }

        try:
          final_themes = chain.invoke(input_data)
          print("Final output:", final_themes)
        except Exception as e:
          print(f"Error occurred while generating final themes: {e}")
          raise

        print(prompt.template.format(
          rqs=self.rqs,
          themes=all_themes,
          format_instructions=format_instructions
      ))

        # Optionally save to a file
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(final_themes, f, indent=4)
                    print(f"Results successfully saved to {filename}")
            elif filename.endswith('.csv'):
                df = pd.DataFrame(final_themes)
                df.to_csv(filename, index=False)
                print(f"Results successfully saved to {filename}")
            else:
                print("Invalid file format. Please use .json or .csv.")

        return final_themes
