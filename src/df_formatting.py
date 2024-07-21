# df_formatting.py

import pandas as pd


class HierarchicalDataFrame:
    def __init__(self, data):
        self.data = data
        self.df = pd.DataFrame(data)
        self.hierarchical_df = None

    def create_hierarchical_index(self):
        # Create MultiIndex for theme, theme_definition, and subthemes
        theme_theme_def_subtheme_pairs = [
            (theme, theme_definition, subtheme)
            for theme, theme_definition, subthemes in zip(self.df['theme'], self.df['theme_definition'], self.df['subthemes'])
            for subtheme in subthemes
        ]

        theme_theme_def_subtheme_index = pd.MultiIndex.from_tuples(theme_theme_def_subtheme_pairs, names=['Theme', 'Theme_Definition', 'Sub-theme'])

        # Flatten the data
        flattened_subtheme_definitions = [
            subtheme_definition
            for subtheme_definitions in self.df['subtheme_definitions']
            for subtheme_definition in subtheme_definitions
        ]

        flattened_supporting_quotes = [
            quote
            for quotes in self.df['supporting_quotes']
            for quote in quotes
        ]

        # Combine subtheme definitions and supporting quotes into a single DataFrame
        hierarchical_data = {
            'Sub-theme Definition': flattened_subtheme_definitions,
            'Supporting Quotes': flattened_supporting_quotes
        }

        self.hierarchical_df = pd.DataFrame(hierarchical_data, index=theme_theme_def_subtheme_index)

    def get_hierarchical_df(self, filename=None, file_format='csv'):
        if self.hierarchical_df is None:
            self.create_hierarchical_index()

        if filename is not None:
            try:
                # Add appropriate file extension if not present
                if file_format == 'csv':
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    self.hierarchical_df.to_csv(filename)
                elif file_format == 'hdf':
                    if not filename.endswith('.h5'):
                        filename += '.h5'
                    self.hierarchical_df.to_hdf(filename, key='df', mode='w')
                print(f"DataFrame successfully saved to {filename}")
            except Exception as e:
                print(f"Error occurred while saving DataFrame: {e}")

        return self.hierarchical_df