# quote_matching.py
import pandas as pd
from fuzzywuzzy import fuzz
from multiprocessing import Pool
from typing import List, Optional, Tuple


class QuoteMatcher:
    def __init__(self, text: List[str], df: pd.DataFrame):
        """
        Initializes the QuoteMatcher with the text and DataFrame.

        Args:
            text (str): The text to match quotes against.
            df (pd.DataFrame): The DataFrame containing the quotes.
        """
        self.text = text.replace('\n', ' ')  # Remove newline characters
        self.df = df

    def _match_quote(self, args: Tuple[str, str, int, int]) -> Optional[Tuple[str, str, int, int]]:
        """
        Helper function to find the best match for a single quote.

        Args:
            args (Tuple[str, str, int, int]): Tuple containing text, quote, index, and threshold.

        Returns:
            Optional[Tuple[str, str, int, int]]: Matched information or None.
        """
        text, quote, index, threshold = args
        ratio = fuzz.partial_ratio(text, quote)
        if ratio > threshold:
            return (text, quote, index, ratio)
        return None

    def quote_matches(self, column: str, threshold: int = 80, filename: Optional[str] = None, use_parallel: bool = False) -> pd.DataFrame:
        """
        Finds fuzzy matches between the given text and quotes in the specified column.

        Args:
            column (str): The column name in the DataFrame to search.
            threshold (int): The similarity threshold for fuzzy matching.
            filename (Optional[str]): Optional csv or hdf filename to save the matched quotes.
            use_parallel (bool): Whether to use parallel processing.

        Returns:
            pd.DataFrame: DataFrame containing matched quotes from the text, matching DataFrame quotes, and their indices.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if self.df.empty:
            print("DataFrame is empty. No matches to find.")
            return pd.DataFrame(columns=['Text', 'Matching DF Quote', 'DF Quote Index', 'Match Ratio'])

        # Split text into sentences or quotes
        text_quotes = self.text.split('. ')

        matches = []

        # Prepare the argument list for matching
        args_list = [(text_quotes, row[column], index, threshold) for text in self.text for index, row in self.df.iterrows()]

        if use_parallel:
            with Pool() as pool:
                results = [pool.apply_async(self._match_quote, (args,)) for args in args_list]
                matches = [result.get() for result in results if result.get() is not None]
        else:
            for args in args_list:
                match = self._match_quote(args)
                if match is not None:
                    matches.append(match)

        print(f"Found {len(matches)} matches out of {len(self.df[column])}")

        # Create a DataFrame from matches
        matched_df = pd.DataFrame(matches, columns=['Text', 'Matching DF Quote', 'DF Quote Index', 'Match Ratio'])

        # Save to file
        if filename is not None:
            try:
                if filename.endswith('.csv'):
                    matched_df.to_csv(filename, index=False)
                elif filename.endswith('.h5'):
                    matched_df.to_hdf(filename, key='df', mode='w')
                else:
                    raise ValueError("Filename must end with .csv or .h5")
                print(f"Results successfully saved to {filename}")
            except Exception as e:
                print(f"Error occurred while saving DataFrame: {e}")

        return matched_df
