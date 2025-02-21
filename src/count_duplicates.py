# count_duplicates.py

from collections import Counter
import re
import json
import pandas as pd


class CountDuplicates:
    """
    Initializes the CountDuplicates with a list of dictionaries.

    Args:
        list_of_dicts (List[Dict]): A list of dictionaries.
        key (str): The key whose values will be checked for duplicates.
    """
    def __init__(self, list_of_dicts, key):
        self.list_of_dicts = list_of_dicts
        self.key = key

    def count_duplicate_strings(self):
        """
        Counts duplicate strings in a list of dictionaries and filters out those with only 1 occurrence.

        Returns:
            dict: A dictionary with strings as keys and their counts as values (only if count > 1).
        """
        # Collect all strings from the specified key in each dictionary
        strings = [d[self.key] for d in self.list_of_dicts if self.key in d]

        # Use Counter to count occurrences and filter out those with count 1
        counted_strings = Counter(strings)

        # Filter out strings with a count of 1
        print(f"Total sum of counts: {counted_strings.total()}")
        return counted_strings

    def filter_dict(self):
        """
        Filters out strings with a count of 1.

        Returns:
            dict: A dictionary with strings as keys and their counts as values (only if count > 1).
        """
        # Use Counter to count occurrences and filter out those with count 1
        counted_strings = self.count_duplicate_strings()

        filtered_dict = {string: count for string, count in counted_strings.items() if count > 1}

        # Print total sum of filtered counts
        print(f"Total sum of filtered counts: {sum(filtered_dict.values())}")

        # Filter out strings with a count of 1
        return filtered_dict

    def top_duplicates(self, top_n = None):
        """
        Return a list of the n most common elements and their counts from the
        most common to the least. If n is omitted or None, most_common() returns
        all elements in the counter. Elements with equal counts will be ordered
        in the order first encountered:

        Args:
            top_n (int): The number of most common elements to return.

        Returns:
            list: A list of the n most common elements and their counts.
        """

        counted_strings = self.count_duplicate_strings()
        return counted_strings.most_common(top_n)

