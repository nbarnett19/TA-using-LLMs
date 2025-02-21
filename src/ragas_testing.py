# ragas_testing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class RAGAsEvaluation:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = self._build_prompt_template()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def _build_rag_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run_inference(self, data_dict: dict) -> dict:
        """
        Runs inference on a dataset where 'answer' and 'contexts' are initially blank.

        Args:
            data_dict (dict): A dictionary with the fields 'question', 'answer', 'contexts', and 'reference'.
                              'answer' and 'contexts' should be blank and will be populated during inference.

        Returns:
            dict: The updated dictionary with answers and contexts filled in.
        """
        rag_chain = self._build_rag_chain()

        # Extracting questions from the dictionary
        questions = data_dict.get("question", [])
        answers = []
        contexts = []

        # Running inference for each question
        for query in questions:
            answers.append(rag_chain.invoke(query))
            contexts.append([doc.page_content for doc in self.retriever.get_relevant_documents(query)])

        # Update the dictionary with the new answers and contexts
        data_dict["answer"] = answers
        data_dict["contexts"] = contexts

        return data_dict

    def evaluate(self, data_dict: dict) -> 'pd.DataFrame':
        dataset = Dataset.from_dict(data_dict)
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        return result.to_pandas()

    def summarize_results(self, results_df: pd.DataFrame, box_title: str = "RAGAs Evaluation Metric Distribution"):
        """
        Summarizes and visualizes the evaluation results with customizable graph titles.

        Args:
            results_df (pd.DataFrame): The DataFrame containing evaluation results.
            bar_title (str): Title for the bar chart. Defaults to "RAGAs Evaluation Metrics".
            box_title (str): Title for the box plot. Defaults to "RAGAs Evaluation Metric Distribution".
        """
        # Summary statistics (mean, median, etc.)
        summary = results_df.describe()

        print("Summary of Results:")
        print(summary)

        # Create visualizations with custom titles
        self._visualize_results(results_df, box_title)

    def _visualize_results(self, results_df: pd.DataFrame, box_title: str):
        """
        Generates bar charts and box plots with custom titles for each evaluation metric.

        Args:
            results_df (pd.DataFrame): The DataFrame containing evaluation results.
            bar_title (str): Title for the bar chart.
            box_title (str): Title for the box plot.
        """
        # sns.set(style="whitegrid")

        # Generate bar plots for each metric
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        results_long = results_df.melt(value_vars=metrics, var_name="Metric", value_name="Score")

        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Metric", y="Score", data=results_long, palette="coolwarm")
        plt.title(box_title)
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

