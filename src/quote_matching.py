# quote_matching.py

import pandas as pd
from fuzzywuzzy import fuzz
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple

class QuoteMatcher:
    def __init__(self, docs, chunks, json_codes_list=None, themes_list=None):
        """
        Initializes the QuoteMatcher with the JSON list and JSON codes list.

        Args:
            docs (List[Dict[str, Any]]):The list of documents to analyze.
            chunks (List[Dict[str, Any]]): The list of text chunks to analyze.
            json_codes_list (List[Dict[str, Any]]): The JSON codes list containing the quotes for matching.
            themes_list (List[Dict[str, Any]]): The JSON themes list containing the quotes for matching.
        """
        self.docs = docs
        self.chunks = chunks
        self.json_codes_list = json_codes_list
        self.themes_list = themes_list

    def matched_theme_quotes(self, threshold=80) -> List[Dict]:
        """
        Finds matched quotes from the list of JSON dictionaries against the content in docs.

        Args:
            threshold (int): The similarity threshold for fuzzy matching.

        Returns:
            List[Dict]: A list of dictionaries with unmatched quotes and their indices.
        """
        quotes = []

        # Check if themes_list is a single dictionary, wrap it in a list
        if isinstance(self.themes_list, dict):
            themes = [self.themes_list]  # Single theme, wrap in list
        else:
            themes = self.themes_list  # Multiple themes

        # Put quotes from themes_list in a list
        for item in themes:
            for quote in item["supporting_quotes"]:
                quotes.append(quote)

        # Total quotes that need to be matched
        print(f"Total number of quotes: {len(quotes)}")

        # Match quotes to chunks
        results = []
        for item in quotes:
            highest_match = None
            highest_ratio = 0

            for chunk in self.chunks:
                match_ratio = fuzz.partial_ratio(item, chunk.page_content)

                if match_ratio > highest_ratio:
                    highest_ratio = match_ratio
                    highest_match = {
                        "quote": item,
                        "matched_chunk": chunk.page_content,
                        "match_ratio": match_ratio,
                        "chunk_id": chunk.metadata.get("source", "unknown")
                    }

                if match_ratio >= threshold:
                    # If the match is above the threshold, add it and stop looking for better matches
                    results.append(highest_match)
                    break
            else:
                # If no match was found above the threshold, add the highest match
                if highest_match:
                    results.append(highest_match)

        print(pd.json_normalize(results))

        return results

    def unmatched_code_excerpts(self, threshold=80):
        """
        Identifies excerpts that do not sufficiently match their corresponding chunk_analyzed fields.

        Args:
            threshold (int): The minimum similarity score required to consider a match (default is 50).

        Returns:
            list of dict: A list of dictionaries with unmatched excerpts, chunks, similarity scores, and their indices.
        """
        unmatched_results = []

        for index, item in enumerate(self.json_codes_list):
            excerpt = item.get("excerpt", "")
            chunk_analyzed = item.get("chunk_analyzed", "")
            score = fuzz.partial_ratio(chunk_analyzed, excerpt)

            if score < threshold:
                unmatched_results.append({
                    "index": index,
                    "code": item.get("code", ""),
                    "excerpt": excerpt,
                    "chunk_analyzed": chunk_analyzed,
                    "match_ratio": score
                })

        if unmatched_results is None:
          print("No unmatched results found.")
        else:
          print(f"{len(unmatched_results)} unmatched results found")

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(unmatched_results)
        print(df)

        return unmatched_results
