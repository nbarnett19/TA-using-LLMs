import json
from fuzzywuzzy import fuzz


class QuoteMatcher:
    def __init__(self, text, json_results):
        self.text = text
        self.json_results = json_results

    def code_excerpt_matches(self, column, threshold=80):
        """
        Finds fuzzy matches between the given text and quotes in the specified column.

        Args:
            column (str): The column name in the DataFrame to search.
            threshold (int): The similarity threshold for fuzzy matching.

        Returns:
            str: A message indicating the number of matches.
        """
        matches = []
        if column in self.json_results.columns:
            for quote in self.json_results[column]:
                if fuzz.partial_ratio(self.text, quote) > threshold:
                    matches.append(quote)
            return f"{len(matches)} out of {len(self.json_results)} quotes match"
        else:
            return f"Column '{column}' not found in the DataFrame."