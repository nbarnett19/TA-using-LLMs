# codes.py

import pandas as pd
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.retrievers import BaseRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


class GenerateCodes(ThematicAnalysis):
    """
    Generates codes and supporting quotes from the text.

    Attributes:
        llm: The language model used to generate responses.
        chunks (str): The text chunks to analyze.
        rqs (str): The research questions to answer.
    """
    def __init__(self, llm, docs, chunks, rqs, examples=None, vector_db=None, retriever=None):
        super().__init__(llm, docs, chunks, rqs)
        self.examples = examples
        self.vector_db = vector_db
        self.retriever = retriever

    def query_transformation(self, template: str, questions: str) -> list:
        """Generates sub-questions from the given research question using decomposition."""

        prompt_transformation = ChatPromptTemplate.from_template(template)
        generate_queries_transformation = (
            prompt_transformation | self.llm | StrOutputParser()
        )

        # Invoke the decomposition chain
        sub_questions = generate_queries_transformation.invoke({"questions": questions})
        return sub_questions

    def generate_codes(self, filename: Optional[str] = None, use_rag: bool = False,
                       rag_query: Optional[str] = None, similarity_search_with_score: bool = False) -> pd.DataFrame:
        """
        Generates codes and supporting quotes from the text, with optional RAG.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.
            use_rag (bool): If True, use RAG to fetch relevant documents before generating codes.
            rag_self_query (Optional[str]): Optional query to use for RAG.
        """

        prompt_codes_template = """You are a qualitative researcher and are doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022). Review the given transcripts
        to identify excerpts (or quotes) that address the research questions.
        Generate codes that best represent each of the excerpts identified. Each
        code should represent the meaning in the excerpt. The excerpts must exactly
        match word for word the text in the transcripts.
        Follow this format {format_instructions}
        Research questions: {rqs}
        The transcripts: {text}
        """
        # Add examples to the template if provided
        if self.examples:
            prompt_codes_template += "Examples: {examples}"

        if use_rag:
            prompt_codes_template = "Context: {context}\n" + prompt_codes_template

        # Ensure JsonOutputParser and format_instructions are correctly defined
        parser = JsonOutputParser(pydantic_object=CodeExcerpt)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_codes_template,
            input_variables=["text", "rqs", "format_instructions", "examples", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_codes = []
        counter = 1

        for data in self.chunks:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing chunk {counter}")
                counter += 1

                # Extract Text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }
                if self.examples:
                    input_data["examples"] = self.examples

                # If RAG is enabled, retrieve relevant documents from the Chroma vector database
                retrieved_docs = []
                if use_rag:
                    original_prompt = prompt.template.format(
                                text=data,
                                rqs=self.rqs,
                                format_instructions=format_instructions,
                                examples=self.examples if self.examples else "",
                                context="")
                    rta_questions = """ How does one perform inductive (latent/semantic)
                    reflexive Thematic analysis according to the book practical guide
                    from Braun and Clark (2022)? How can one identify excerpts (or quotes)
                    that address the research questions in reflexive thematic analysis? """
                    if rag_query is not None:
                        # Decompose research questions into sub-questions
                        sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions + text)
                        results = self.retriever.invoke(sub_questions)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                    elif similarity_search_with_score==True:
                        results = self.vector_db.similarity_search_with_score(original_prompt)
                        for doc, score in results:
                           retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                    else:
                        results = self.retriever.invoke(original_prompt)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

                    # Combine the retrieved documents with the original chunk text
                    if similarity_search_with_score:
                      input_data["context"] = "\n".join([doc.page_content for doc, _ in results])
                    else:
                      input_data["context"] = "\n".join([doc.page_content for doc in results])
                    print(f"Retrieved documents: {retrieved_docs}")

                # Generate codes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a single dictionary, convert it to a list of one item
                if isinstance(response, dict):
                  response = [response]

                # If response is a list, use it directly
                if isinstance(response, list):
                  codes = response

                  # Ensure codes are in the expected format
                  for code in codes:
                      if not isinstance(code, dict) or not all(key in code for key in ["code", "excerpt", "speaker"]):
                          raise ValueError("Invalid code format detected.")

                      # Fill missing values
                      code['chunk_analyzed'] = text
                      code['source'] = source_file  # Add source file information
                      if rag_query is not None:
                          code['RAG_query'] = sub_questions
                          code['retrieved_documents'] = retrieved_docs
                      elif use_rag:
                          code['retrieved_documents'] = retrieved_docs
                      else:
                          continue

                  all_codes.extend(codes)  # Flatten the results
                else:
                    raise ValueError("Unexpected response format from model.")

            except Exception as e:
                print(f"Error occurred while processing chunk {counter} in {source_file}: {e}")

        try:
            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(all_codes)
            print(f"DataFrame shape: {df.shape}")
            print(prompt.template.format(
                            text=chunks,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context=retrieved_docs if use_rag else ""
                        ))  # Add the prompt
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(all_codes, f, indent=4)
                        print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return all_codes

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")
            raise

    def cot_coding(self, filename: Optional[str] = None, use_rag: bool = False,
                   rag_query: Optional[str] = None, similarity_search_with_score: bool = False):
        """
        Generates codes and supporting quotes from the text.

        """
        cot_prompt_template = """
        You are a qualitative researcher and are doing inductive (latent/semantic)
        reflexive Thematic analysis according to the book practical guide from
        Braun and Clark (2022).
        Follow these steps to analyze the transcripts:

        1. **Review the Transcripts:** Carefully read the transcripts to identify
        key excerpts related to the research questions.

        2. **Generate Codes:** Generate codes that best represent each of the excerpts identified.
        Each code should represent the meaning in the excerpt. Codes should be a mix of
        semantic and latent codes. Semantic means the analysis captures meanings
        that are explicitly stated in the data, so that words themselves are taken at face value.
        Latent means the analysis captures meanings not explicitly stated in the data,
        including the ideas, assumptions, or concepts that underpin what is explicitly stated.

        3. **Match Excerpts:** For each code, find exact excerpts from the transcripts that support it.

        4. **Describe the code:** For each code, describe its meaning in the excerpt.

        4. **Organize the Results:** Format your findings according to the following: {format_instructions}.

        Example Research Questions:
        What are educators’ general attitudes toward the promotion of student wellbeing
        and towards a set of ‘wellbeing guidelines’ recently introduced in Irish
        post-primary schools. What are the potential barriers to wellbeing promotion
        and what are educators’ opinions as to what might constitute opposite remedial
        measures in this regard?

        **Example Transcript:**
        P1: I think anything that you do in school that's on paper is difficult
        to relate to students. And, this is the great thing about the new junior-cycle,
        there's a lot more of the hands on approach in most academic subjects.
        I think, that needs to be brought into areas like SPHE. Theory is fine -
        I don't know if you want me to talk about the wellbeing indicators
        [interviewer gestures to continue]. I have them there on my wall, this is
        maybe my third year to have them on my wall. To be honest, I feel that that's
        just way too abstract!
        P2: Although the hands-on approach simulates real life scenarios, I find
        that thoroughly teaching the theory provides students with the tools they
        need to be successful. And some of my students prefer this approach.

        **Example Output:**
        Code name: The wellbeing curriculum is not relatable for the students
        Code description: The participant noted the difficulty students have in relating to school curricula.,
        Excerpt: I think that anything that you do in school that's on paper is difficult to relate to students.

        Code name: A practical approach to learning is beneficial for students
        Code description: The participant praised the hands on approach in the new junior-cycle
        Excerpt: And, this is the great thing about the new junior-cycle, there's
        a lot more of the hands on approach in most academic subjects.

        Code name: Wellbeing promotion should be practical
        Code description: The participant felt there is a need to bring practical approaches into SPHE
        Excerpt: I think, that needs to be brought into areas like SPHE.

        Code name: The wellbeing guidelines lack clarity!
        Code description: The participant emphasized how abstract they found the current written guidelines.
        Excerpt: To be honest, I feel that that's just way too abstract!

        Code name: Theoretical approach is necessary for wellbeing promotion success
        Code description: Participant preferred the theoretical approach to well-being
        education as it fit some students’ learning styles.
        Excerpt: I find that thoroughly teaching the theory provides students with the tools they need to be successful.

        Now, apply this process to the provided transcripts.
        transcript: {text}
        research questions: {rqs}
        """

        if use_rag:
            cot_prompt_template = "Context: {context}\n" + cot_prompt_template

        # Ensure JsonOutputParser and format_instructions are correctly defined
        parser = JsonOutputParser(pydantic_object=CodeExcerpt)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=cot_prompt_template,
            input_variables=["text", "rqs", "format_instructions", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        all_codes = []
        counter = 1

        for data in self.chunks:
            try:
                source_file = data.metadata.get("source", "Unknown")
                print(f"Processing chunk {counter}")
                counter += 1

                # Extract Text
                text = data.page_content

                # Prepare input dictionary
                input_data = {
                    "rqs": self.rqs,
                    "text": text
                }

                # If RAG is enabled, retrieve relevant documents from the Chroma vector database
                retrieved_docs = []
                if use_rag:
                    original_prompt = prompt.template.format(
                                text=data,
                                rqs=self.rqs,
                                format_instructions=format_instructions,
                                examples=self.examples if self.examples else "",
                                context="")
                    rta_questions = """ How does one perform inductive (latent/semantic)
                    reflexive Thematic analysis according to the book practical guide
                    from Braun and Clark (2022)? How can one identify excerpts (or quotes)
                    that address the research questions in reflexive thematic analysis? """
                    if rag_query is not None:
                        # Decompose research questions into sub-questions
                        sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions + text)
                        results = self.retriever.invoke(sub_questions)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")
                    elif similarity_search_with_score==True:
                        results = self.vector_db.similarity_search_with_score(original_prompt)
                        for doc, score in results:
                           retrieved_docs.append(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
                    else:
                        results = self.retriever.invoke(original_prompt)
                        for doc in results:
                           retrieved_docs.append(f"* {doc.page_content} [Source: {doc.metadata['source']}]")

                    # Combine the retrieved documents with the original chunk text
                    if similarity_search_with_score:
                      input_data["context"] = "\n".join([doc.page_content for doc, _ in results])
                    else:
                      input_data["context"] = "\n".join([doc.page_content for doc in results])
                    print(f"Retrieved documents: {retrieved_docs}")

                # Generate codes
                response = chain.invoke(input_data)
                print(f"Model output: {response}")

                # If response is a single dictionary, convert it to a list of one item
                if isinstance(response, dict):
                  response = [response]

                # If response is a list, use it directly
                if isinstance(response, list):
                  codes = response

                  # Ensure codes are in the expected format
                  for code in codes:
                      if not isinstance(code, dict) or not all(key in code for key in ["code", "excerpt", "speaker"]):
                          raise ValueError("Invalid code format detected.")

                      # Fill missing values
                      code['chunk_analyzed'] = text
                      code['source'] = source_file  # Add source file information
                      if rag_query is not None:
                          code['RAG_query'] = sub_questions
                          code['retrieved_documents'] = retrieved_docs
                      elif use_rag:
                          code['retrieved_documents'] = retrieved_docs
                      else:
                          continue

                  all_codes.extend(codes)  # Flatten the results
                else:
                    raise ValueError("Unexpected response format from model.")

            except Exception as e:
                print(f"Error occurred while processing chunk {counter} in {source_file}: {e}")

        try:
            # Convert the flattened list of dictionaries to a DataFrame
            df = pd.DataFrame(all_codes)
            print(f"DataFrame shape: {df.shape}")
            print(prompt.template.format(
                            text=chunks,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            context=retrieved_docs if use_rag else ""
                        ))  # Add the prompt
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(all_codes, f, indent=4)
                        print(f"Results successfully saved to {filename}")
                elif filename.endswith('.csv'):
                    df.to_csv(filename, index=False)
                    print(f"Results successfully saved to {filename}")
                else:
                    print("Invalid file format. Please use .json or .csv.")
            return all_codes

        except Exception as e:
            print(f"Error occurred while converting JSON to DataFrame: {e}")
            raise