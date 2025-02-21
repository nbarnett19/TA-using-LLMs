# themes.py

class GenerateThemes:
    """
    Generates themes and supporting quotes from the codes list.

    Attributes:
        llm: The language model used to generate responses.
        rqs (str): The research questions to answer.
        json_codes_list (List[Dict[str, Any]]): List of codes containing generated codes and their details.
        examples (Optional[List[str]]): Optional list of examples to include in the prompt.
    """
    def __init__(self, llm, rqs, json_codes_list, examples=None, vector_db=None, retriever=None):
        self.llm = llm
        self.rqs = rqs
        self.json_codes_list = json_codes_list
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

    def generate_themes(self, filename = None, use_rag: bool = False,
                        rag_query = None, similarity_search_with_score: bool = False):
        """
        Generates themes with definitions and supporting quotes from the codes list.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """
        prompt_codes_template = """You are a qualitative researcher and are doing
        inductive (latent/semantic) reflexive Thematic analysis according to the
        book practical guide from Braun and Clark (2022).
        Based on the research questions provided, you need to collate the codes
        into a maximum of 6 distinct themes. Each theme should include:
        1. A theme definition including a title
        2. Sub-themes if needed
        3. Each sub-theme should have a definition
        4. 2 supporting quotes for each theme and subtheme
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.

        Codes: {codes}
        Follow this format: {format_instructions}.
        Research questions: {rqs} """

        # Add examples to the template if provided
        if self.examples:
            prompt_codes_template += "Examples: {examples}"
        # Add context if use_rag = true
        if use_rag:
            prompt_codes_template = "Context: {context}" + prompt_codes_template

        parser = JsonOutputParser(pydantic_object=Themes)
        format_instructions = parser.get_format_instructions()

        # Filter json data to only necessary fields
        fields_to_keep = ["code", "code_description", "excerpt"]

        # Create a new list to store filtered data
        filtered_data = []
        for item in self.json_codes_list:
            filtered_item = {field: item.get(field) for field in fields_to_keep} # Extract specified fields from each dictionary
            filtered_data.append(filtered_item)

        prompt = PromptTemplate(
            template=prompt_codes_template,
            input_variables=["codes", "rqs", "format_instructions", "examples", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        try:
            # Prepare input dictionary
            input_data = {
              "codes": filtered_data,
              "rqs": self.rqs
              }
              # Add examples to the template if provided
            if self.examples:
                input_data["examples"] = self.examples

            # If RAG is enabled, retrieve relevant documents from the Chroma vector database
            retrieved_docs = []
            if use_rag:
                original_prompt = prompt.template.format(
                            codes=filtered_data,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context="")
                rta_questions = """ How does one perform inductive (latent/semantic)
                reflexive Thematic analysis according to the book practical guide
                from Braun and Clark (2022)? How can one collate codes into themes
                that address the research questions in reflexive thematic analysis?
                When defining the themes and subthemes, what does it mean to look for data
                (codes, quotations) that contradict or are discrepant to the established
                themes and subthemes? """
                if rag_query is not None:
                    # Decompose research questions into sub-questions
                    sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions)
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
            if use_rag:
              input_data["context"] = retrieved_docs

            # Generate themes
            results = chain.invoke(input_data)
            print(prompt.template.format(
                    codes=filtered_data,
                    rqs=self.rqs,
                    format_instructions=format_instructions,
                    examples=self.examples if self.examples else "",
                    context=retrieved_docs if use_rag else ""
                ))
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
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
            print(f"Error occurred while processing themes: {e}")
            raise

    def cot_themes(self, filename = None, use_rag: bool = False,
                   rag_query = None, similarity_search_with_score: bool = False):
        """
        Generates themes with definitions and supporting quotes from the codes list.

        Args:
            filename (Optional[str]): Optional filename to save the generated themes.

        Returns:
            The generated response from the language model.
        """
        cot_theme_template = """
        Objective: You are a qualitative researcher and are doing inductive
        (latent/semantic) reflexive Thematic analysis according to the book practical
        guide from Braun and Clark (2022).

        Steps:
        1. Group codes into subthemes: Organize related codes into subthemes, if needed, that
        capture shared meanings across the codes based on the research questions provided.
        When subthemes are present, provide a definition for each subtheme.

        2. Group subthemes into themes: Organize related subthemes (if present) or codes into a
        maximum of 6 distinct themes that capture shared meanings across the subthemes
        based on the research questions provided. A subtheme sits under a theme.
        It focuses on one particular aspect of that theme; it brings analytic attention
        and emphasis on this aspect. Use subthemes only when they are needed to
        bring emphasis to one particular aspect of a theme.
                                                                                                                                                                                                                                                                               Support each theme and subtheme (if needed) with at least 2 supporting quotes.
        3. Provide a clear definition for each theme, showing how it addresses
        the research questions. In case you have subthemes do the same with these (definition).
        When defining the themes and subthemes, please look for data (codes, quotations)
        that contradict or are discrepant to the – so far- established themes and subthemes.
        Please use these contradictory data to either refine themes or subthemes
        or add new themes or subthemes.

        4. Present Findings: Use this format: {format_instructions}.

        ### Example Analysis:

        **Example Research Questions:**
        What are educators’ general attitudes toward the promotion of student wellbeing
        and towards a set of ‘wellbeing guidelines’ recently introduced in Irish
        post-primary schools. What are the potential barriers to wellbeing promotion
        and what are educators’ opinions as to what might constitute opposite remedial measures in this regard?

        **Example Codes:**
        Code name: The wellbeing curriculum is not relatable for the students
        Code description: The participant noted the difficulty students have in relating to school curricula.

        Code name: A practical approach to learning is beneficial for students
        Code description: The participant praised the hands on approach in the new junior-cycle

        Code name: Wellbeing promotion should be practical
        Code description: The participant felt there is a need to bring practical approaches into SPHE

        Code name: The wellbeing guidelines lack clarity
        Code description": The participant emphasized how abstract they found the current guidelines.

        Code name: Wellbeing promotion requires involvement from all staff members
        Code description: The participant stressed that effective wellbeing promotion
        demands active participation from all staff members, not just a select few.

        Code name: Staff collaboration enhances student wellbeing outcomes
        Code description: The participant highlighted the importance of collaboration
        among school staff in ensuring positive wellbeing outcomes for students.

        Code name: School leadership plays a crucial role in driving wellbeing initiatives
        Code description: The participant emphasized that school leadership is
        key to implementing and sustaining successful wellbeing promotion efforts.

        Code name: Theoretical approach is necessary for wellbeing promotion success
        Code description: Participants preferred the theoretical approach to well-being
        education as it fit some students’ learning styles.

        **Example Output:**
        Theme: An integrative approach to wellbeing promotion
        Theme definition: This theme captures two distinct yet complementary approaches
        to enhancing wellbeing promotion within schools. One narrative emphasizes
        the collective responsibility of the entire school staff in fostering student
        wellbeing, while the other focuses on taking students learning preferences into
        account with the majority of  students preferring a practical, hands-on approach
        for effective wellbeing promotion. Together, these sub-themes represent two
        independently valuable perspectives on how best practices can be applied to
        create meaningful, actionable outcomes in wellbeing initiatives.

        Subthemes:
        Subtheme: Taking student learning preferences into account with the delivery of wellbeing promotion
        Subtheme definition: Many participants highlighted the need for practical wellbeing
        promotion, however, there were some discrepant opinions which suggest a theoretical
        base is still considered necessary
        Relevant codes: A practical approach to learning is beneficial for students,
        Wellbeing promotion should be practical, The wellbeing guidelines lack clarity,
        The wellbeing curriculum is not relatable for the students, Theoretical approach is
        necessary for wellbeing promotion success

        Subtheme: The Whole-School Approach
        Subtheme definition: This subtheme emphasizes the importance of involving
        all members of the school community in promoting student wellbeing. Participants
        stressed that wellbeing should not be confined to a single department or role,
        but rather integrated throughout the entire school.
        Relevant codes: Wellbeing promotion requires involvement from all staff members,
        Staff collaboration enhances student wellbeing outcomes, School leadership
        plays a crucial role in driving wellbeing initiatives

        Now, apply this process to the provided codes, ensuring that each step is followed meticulously.
        Your final output should include a maximum list of 6 themes and subthemes
        if needed, each with their respective definitions and supporting quotes
        that accurately reflect the data.

        codes: {codes}
        research questions: {rqs}
        """
        # Add context if use_rag = true
        if use_rag:
            cot_theme_template = "Context: {context}" + cot_theme_template

        parser = JsonOutputParser(pydantic_object=Themes)
        format_instructions = parser.get_format_instructions()

        # Filter json data to only necessary fields
        fields_to_keep = ["code", "code_description", "excerpt"]

        # Create a new list to store filtered data
        filtered_data = []
        for item in self.json_codes_list:
            filtered_item = {field: item.get(field) for field in fields_to_keep} # Extract specified fields from each dictionary
            filtered_data.append(filtered_item)

        prompt = PromptTemplate(
            template=cot_theme_template,
            input_variables=["codes", "rqs", "format_instructions"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | parser

        try:
            # Prepare input dictionary
            input_data = {
              "codes": filtered_data,
              "rqs": self.rqs
              }

            # If RAG is enabled, retrieve relevant documents from the Chroma vector database
            retrieved_docs = []
            if use_rag:
                original_prompt = prompt.template.format(
                            codes=filtered_data,
                            rqs=self.rqs,
                            format_instructions=format_instructions,
                            examples=self.examples if self.examples else "",
                            context="")
                rta_questions = """ How does one perform inductive (latent/semantic)
                reflexive Thematic analysis according to the book practical guide
                from Braun and Clark (2022)? How can one collate codes into themes
                that address the research questions in reflexive thematic analysis?
                When defining the themes and subthemes, what does it mean to look for data
                (codes, quotations) that contradict or are discrepant to the established
                themes and subthemes? """
                if rag_query is not None:
                    # Decompose research questions into sub-questions
                    sub_questions = self.query_transformation(template=rag_query, questions=self.rqs + rta_questions)
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
            if use_rag:
              input_data["context"] = retrieved_docs

            # Generate themes
            results = chain.invoke(input_data)
            print(prompt.template.format(
                    codes=filtered_data,
                    rqs=self.rqs,
                    format_instructions=format_instructions,
                    context=retrieved_docs if use_rag else ""
                ))
            if rag_query is not None:
              print(f"Sub_questions: {sub_questions}")

            # Save results to file
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
            print(f"Error occurred while processing themes: {e}")
            raise