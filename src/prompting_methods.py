# prompting_methods.py

import json
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from JSONparser import CodeExcerpt, ZSThemes


class ThematicAnalysis:
    """
    Generates themes with definitions and supporting quotes from the text.

    Attributes:
        llm: The language model used to generate responses.
    """

    def __init__(self, llm, chunks, rqs):
        self.llm = llm
        self.chunks = chunks
        self.rqs = rqs

    def generate_summary(self):
        """
        Generates summary from the text based on research questions.

        Args:
            rqs (str): The research questions to answer.

        Returns:
            The generated response from the language model for the combined text.
        """
        template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions, \
        generate a summary from the text: {text}."""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm

        summaries = []
        for chunk in self.chunks:
            try:
                # Generate summary for each chunk
                summary = chain.invoke({"rqs": self.rqs, "text": chunk})
                summaries.append(summary.content)
            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        # Combine all chunk summaries into a single summary
        combined_summary_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions and the following summaries, \
        generate an overall summary: {summaries}."""
        combined_summary_prompt = PromptTemplate.from_template(combined_summary_template)
        combined_chain = combined_summary_prompt | self.llm

        try:
            final_summary = combined_chain.invoke({"rqs": self.rqs, "summaries": "\n\n".join(summaries)})
            print("Final summary:", final_summary.content)
            return final_summary.usage_metadata
        except Exception as e:
            print(f"Error occurred while generating final summary: {e}")
            raise

    def generate_codes(self, min_codes=5, max_codes=10, filename=None):
        """
        Generates codes for the text based on research questions.

        Args:
            min_codes (int): Minimum number of codes to generate.
            max_codes (int): Maximum number of codes to generate.
            filename (str): Optional filename to save the generated codes.

        Returns:
            DataFrame of generated codes.
        """
        prompt_codes_template = """You are a qualitative researcher. \
        Review the given transcripts to identify relevant excerpts that address the research question.
        Generate between {min_codes} and {max_codes} phrases (or codes) that best represent the excerpts identified. \
        Each code must be between two to five words long.
        <format_instructions>
        {format_instructions}
        Where code1 and code2 are the codes you generated and excerpt1 and excerpt2 are the excerpts that support the code.
        </format_instructions>

        <transcript>
        {text}
        </transcript>

        <question>
        {rqs}
        </question>

        codes:"""

        # Set up the output parser
        parser = JsonOutputParser(pydantic_object=CodeExcerpt)

        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_codes_template,
            input_variables=["text", "rqs", "min_codes", "max_codes", "format_instructions"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_codes = []
        for chunk in self.chunks:
            try:
                # Generate codes for each chunk
                codes = chain.invoke({
                    "rqs": self.rqs,
                    "text": chunk,
                    "min_codes": min_codes,
                    "max_codes": max_codes,
                })

                # Append the codes to the list
                all_codes.append(codes)

            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

        try:
            # Flatten the list of lists
            flat_results = [item for sublist in all_codes for item in sublist]

            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(flat_results)

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")

        if filename is not None:
            try:
                with open(filename, 'w') as f:
                    json.dump(flat_results, f, indent=4)

                    print(f"Results successfully saved to {filename}")

            except Exception as e:
                print(f"Error occurred while saving JSON data: {e}")

        return df

    def zs_prompt(self, codes_df, filename=None):
        """
        Generates themes with definitions and supporting quotes from the text.

        Args:
            codes_df (DataFrame): DataFrame containing the generated codes.

        Returns:
            The generated response from the language model.
        """
        zs_template = """You are a qualitative researcher. \
        The aim of your study is to answer the following research questions: {rqs} \
        Based on your research questions, collate the codes into themes, theme \
        definitions, subthemes, subtheme definitions and supporting quotes
        <format_instructions>
        {format_instructions}
        Where theme1 and theme2 are the themes you generated and definition1 and definition2 \
        are the definitions of the themes. Subtheme1 and subtheme2 are the subthemes of the theme. \
        and subtheme_definition1 and subtheme_definition2 are the definitions of the subthemes. \
        Supporting_quote1 and supporting_quote2 are the supporting quotes for the theme.
        </format_instructions>

        <codes>
        {codes}
        </codes>

        <question>
        {rqs}
        </question>

        Thematic Analysis:"""""

        # Convert the DataFrame to a list of dictionaries
        codes = codes_df.to_dict(orient='records')

        # Set up the output parser
        parser = JsonOutputParser(pydantic_object=ZSThemes)

        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=zs_template,
            input_variables=["codes", "rqs", "format_instructions"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_themes = []
        for code in codes:
            try:
                # Generate codes for each chunk
                themes = chain.invoke({
                    "rqs": self.rqs,
                    "codes": codes,
                })

                # Append the codes to the list
                all_themes.append(themes)

            except Exception as e:
                print(f"Error occurred while processing code: {e}")

        try:
            # Flatten the list of lists
            flat_results = [item for sublist in all_themes for item in sublist]

            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(flat_results)

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")

        if filename is not None:
            try:
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)

                    print(f"Results successfully saved to {filename}")

            except Exception as e:
                print(f"Error occurred while saving JSON data: {e}")

        return df

