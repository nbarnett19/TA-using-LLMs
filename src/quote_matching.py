# quote_matching.py
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from multiprocessing import Pool


class QuoteMatcher:
    def __init__(self, text, df):
        """
        Initializes the QuoteMatcher with the text and DataFrame.

        Args:
            text (str): The text to match quotes against.
            df (pd.DataFrame): The DataFrame containing the quotes.
        """
        self.text = text.replace('\n', ' ')  # Remove newline characters
        self.df = df

    def _match_quote(self, text_quote, column, threshold):
        """
        Helper function to find the best match for a single quote.
        """
        best_match = None
        highest_ratio = 0
        best_match_index = None
        for index, row in self.df.iterrows():
            quote = row[column]
            ratio = fuzz.partial_ratio(text_quote, quote)  # Use a potentially faster matching method
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = quote
                best_match_index = index
        if highest_ratio > threshold:
            return (text_quote, best_match, best_match_index, highest_ratio)
        return None

    def quote_matches(self, column, threshold=80, filename=None, use_parallel=False):
        """
        Finds fuzzy matches between the given text and quotes in the specified column.

        Args:
            column (str): The column name in the DataFrame to search.
            threshold (int): The similarity threshold for fuzzy matching.
            filename (str): Optional csv or hdf filename to save the matched quotes.
            use_parallel (bool): Whether to use parallel processing.

        Returns:
            pd.DataFrame: DataFrame containing matched quotes from the text, matching DataFrame quotes, and their indices.
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found in the DataFrame."

        # Split text into sentences or quotes
        text_quotes = self.text.split('. ')  # Adjust split as per your needs

        if use_parallel:
            with Pool() as pool:
                matches = pool.starmap(self._match_quote, [(quote, column, threshold) for quote in text_quotes])
        else:
            matches = [self._match_quote(quote, column, threshold) for quote in text_quotes]

        # Filter out None values
        matches = [match for match in matches if match]

        # Create a DataFrame from matches
        matched_df = pd.DataFrame(matches, columns=['Text Quote', 'Matching JSON Quote', 'JSON Quote Index', 'Match Ratio'])

        # Save to file
        if filename is not None:
            try:
                if filename.endswith('.csv'):
                    matched_df.to_csv(filename, index=False)
                elif filename.endswith('.h5'):
                    matched_df.to_hdf(filename, key='df', mode='w')
                print(f"Results successfully saved to {filename}")
            except Exception as e:
                print(f"Error occurred while saving DataFrame: {e}")

        return matched_df
