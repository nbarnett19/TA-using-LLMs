# quote_matching.py

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


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

    def code_excerpt_matches(self, column, threshold=80, filename=None):
        """
        Finds fuzzy matches between the given text and quotes in the specified column.

        Args:
            column (str): The column name in the DataFrame to search.
            threshold (int): The similarity threshold for fuzzy matching.
            filename (str): Optional csv or hdf filename to save the matched quotes.

        Returns:
            pd.DataFrame: DataFrame containing matched quotes from the text, matching DataFrame quotes, and their indices.
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found in the DataFrame."

        # Split text into sentences or quotes
        text_quotes = self.text.split('. ')  # Adjust split as per your needs

        matches = []
        for text_quote in text_quotes:
            best_match = None
            highest_ratio = 0
            best_match_index = None
            for index, row in self.df.iterrows():
                quote = row[column]
                ratio = fuzz.partial_ratio(text_quote, quote)
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = quote
                    best_match_index = index
            if highest_ratio > threshold:
                matches.append((text_quote, best_match, best_match_index))

        # Create a DataFrame from matches
        matched_df = pd.DataFrame(matches, columns=['Text Quote', 'Matching JSON Quote', 'JSON Quote Index'])

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
