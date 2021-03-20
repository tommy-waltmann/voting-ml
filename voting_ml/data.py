import numpy as np
import pandas as pd


class PollDataProxy:
    """
    This class acts as a proxy to all of the data in the poll used for this
    assignment. It uses a pandas dataframe object to allow access to the answers
    to the poll questions as properties of the class. Ideally all the pandas in
    this project will appear in this file and only in this file.
    """

    def __init__(self, remove_nan=False):
        self._remove_nan = remove_nan
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

    def all_data(self, question_list=None):
        """
        Get all the responses for the specified poll questions.

        Args:
            question_list (list(str)):
                A list of questions to get the response data for, defaults to
                all questions.

        Returns (np.ndarray(N_answers, N_questions)):
            All of the answers for the specified questions in the poll.
        """
        if question_list is None:
            question_list = self.questions()

        return_table = np.zeros((self._dataframe.shape[0], len(question_list)))
        for i, question in enumerate(question_list):
            # get column index, throw exception if question is not valid
            try:
                answers = self._dataframe[question]
            except KeyError:
                raise KeyError("{} is not a valid question in the poll".format(question))

            # get the data from the table and add it to the result
            if question in self._cols_with_string_answers:
                return_table[:, i] = self._answers_as_ints(answers, question)
            else:
                return_table[:, i] = answers

        # remove all rows with a nan value in them
        if self._remove_nan:
            return_table = return_table[~np.isnan(return_table).any(axis=1)]

        return return_table

    def questions(self):
        """
        Get a list of all the questions in the poll.

        Returns (list(str)):
            A list of the numbers for each question in the poll.
        """
        return list(self._dataframe.columns)

    def _answers_as_ints(self, responses, column_name):
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

        # get the answers to the question as int
        question_answers = self._dataframe[column_name]
        if column_name in self._cols_with_string_answers:
            result = self._answers_as_ints(question_answers, column_name)
        elif column_name in cols:
            result = np.array(question_answers)
        else:
            raise RuntimeError("Cannot find column for {} in the poll data".format(column_name))

        # remove nan values
        if self._remove_nan:
            result = result[np.logical_not(np.isnan(result))]

        return result
