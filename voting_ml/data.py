import numpy as np
import pandas as pd


class PollDataProxy:
    """
    This class acts as a proxy to all of the data in the poll used for this
    assignment. It uses a pandas dataframe object to allow access to the answers
    to the poll questions as properties of the class. Ideally all the pandas in
    this project will appear in this file and only in this file.
    """

    def __init__(self):
        self._dataframe = pd.read_csv(
            "../extern/fivethirtyeight_data/non-voters/nonvoters_data.csv"
        )

        # mapping the questions that have string answers to their keys
        self._string_int_mapping = {
            'race': {'White': 0, 'Black': 1, 'Other/Mixed': 2, 'Hispanic': 3},
            'income_cat': {'Less than $40k': 0, '$40-75k': 1, '$75-125k': 2, '$125k or more': 3},
            'gender': {'Female': 0, 'Male': 1},
            'voter_category': {'rarely/never': 0, 'sporadic': 1, 'always': 2},
            'educ': {'High school or less': 0, 'Some college': 1, 'College': 2}
        }

        # columns with string answers
        self._cols_with_string_answers = list(self._string_int_mapping.keys())

    def answers_as_ints(self, responses, column_name):
        """
        Convert all the string response data to the ints that correspond to
        those answers.

        Args:
            responses (lst(str)):
                All the responses to the given question.
            column_name (str):
                Name of the question that the responses correspond to.
        """
        mapping_dict = self._string_int_mapping[column_name]
        responses_as_ints = np.zeros(len(responses))
        for i, string_response in enumerate(responses):
            responses_as_ints[i] = mapping_dict[string_response]
        return responses_as_ints

    def __getattr__(self, column_name):
        """ Get the answers to each poll question as lists. """
        cols = self._dataframe.columns
        if column_name in self._cols_with_string_answers:
            return self.answers_as_ints(self._dataframe[column_name], column_name)
        elif column_name in cols:
            return np.array(self._dataframe[column_name])
        else:
            raise RuntimeError(
                "Cannot find column for {} in the poll data".format(column_name))
