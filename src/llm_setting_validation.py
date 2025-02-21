# llm_setting_validation.py

import nltk
from nltk import ngrams as nltk_ngrams
from typing import Optional, List
import json
import pandas as pd

nltk.download('punkt')


class LLMTextDiversityAnalyzer:
    def __init__(self, thematic_analysis):
        """
        Initializes the class and sets the thematic analysis instance.

        Args:
            thematic_analysis: Instance of ThematicAnalysis to generate themes and codes.
        """
        self.thematic_analysis = thematic_analysis  # ThematicAnalysis instance

    def run_thematic_analysis(self, runs=10, filename: Optional[str] = None):
        """
        Executes the thematic analysis by calling the zs_codes method.

        Args:
            runs (int): Number of times to run the thematic analysis.
            filename: Optional filename to save the thematic analysis result.

        Returns:
            The result of the thematic analysis.
        """
        codes = []
        for i in range(runs):
            try:
                codes.append(self.thematic_analysis.generate_codes())
                print(f"Thematic analysis {i+1} successfully run.")
            except Exception as e:
                print(f"Error running thematic analysis {i}: {e}")
                raise
        # Save results to file
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(codes, f, indent=4)
                print(f"Results successfully saved to {filename}")
        self.zs_code_results = codes
        return codes

    def set_code_data(self):
        """
        Extracts relevant theme data (theme, subthemes, codes) from the zs_control_gemini output.
        """
        if not hasattr(self, 'zs_code_results'):
            raise ValueError("Thematic analysis has not been run yet. Run run_thematic_analysis() first.")

        # t5p5[0][0]['code']
        all_runs = []
        for result in self.zs_code_results:
          all_codes = ""
          # Access the elements within the nested structure using their appropriate index
          for i in result:
            all_codes += i['code'] + " "
          all_runs.append(all_codes)
        self.all_runs = all_runs
        return all_runs

    def count_tokens(self):
        """Counts the number of tokens in a text."""
        all_tokens = []
        num_of_tokens = []
        for i, run in enumerate(self.all_runs):
            tokens = nltk.word_tokenize(run)
            all_tokens.append(tokens)
            num_of_tokens.append(len(tokens))
            print(f"Run {i + 1} token count: {len(tokens)}")
        self.all_tokens = all_tokens
        self.num_of_tokens = num_of_tokens
        return all_tokens, num_of_tokens

    def count_unique_ngrams(self, n=2):
        """Counts the unique n-grams (bi-grams, tri-grams, etc.) in a text."""
        ngrams = []
        unique_ngram_count = []
        for i, tokens in enumerate(self.all_tokens):
            n_grams = list(nltk_ngrams(tokens, n))
            ngrams.append(n_grams)
            unique_ngrams = set(n_grams)
            unique_ngram_count.append(len(unique_ngrams))
            print(f"Unique {n}-grams in run {i + 1}: {len(unique_ngrams)}")
        if n == 2:
            self.bigrams = ngrams
            self.unique_bigram_count = unique_ngram_count
        elif n == 3:
            self.trigrams = ngrams
            self.unique_trigram_count = unique_ngram_count
        else:
            self.ngrams = ngrams
            self.unique_ngram_count = unique_ngram_count
        return ngrams, unique_ngram_count

    def display_results(self):
        """Displays diversity metrics."""
        df = pd.DataFrame({
            "Tokens": self.all_tokens,
            "Token Count": self.num_of_tokens,
            "Bigrams": self.bigrams,
            "Unique Bigrams": self.unique_bigram_count,
            "Trigrams": self.trigrams,
            "Unique Trigrams": self.unique_trigram_count
        })

        print(df.describe())
        return df

