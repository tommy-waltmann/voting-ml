import numpy as np
import pandas as pd


class PollDataProxy:
    """
    This class acts as a proxy to all of the data in the poll used for this
    assignment. It uses a pandas dataframe object to allow access to the answers
    to the poll questions as properties of the class. Ideally all the pandas in
    this project will appear in this file and only in this file.

    Args:
        remove_nan (bool):
            Whether values returned by this instance should include empty answers
            to the questions, defaults to False.
        convert_to_float (bool):
            Whether values returned by this instance should convert string answers
            to the value of their numerical mapping, defaults to False.
    """

    def __init__(self, remove_nan=False, convert_to_float=False):
        self._remove_nan = remove_nan
        self._convert_to_float = convert_to_float
        self._dataframe = pd.read_csv(
            "../extern/fivethirtyeight_data/non-voters/nonvoters_data.csv"
        )
        self._list_all_question = self.questions()
        self._total_questions = len(self.questions())
        # mapping the questions that have string answers to their keys
        self._string_int_mapping = {
            'race': {'White': 0, 'Black': 1, 'Other/Mixed': 2, 'Hispanic': 3},
            'income_cat': {'Less than $40k': 0, '$40-75k': 1, '$75-125k': 2, '$125k or more': 3},
            'gender': {'Female': 0, 'Male': 1},
            'voter_category': {'rarely/never': 0, 'sporadic': 1, 'always': 2},
            'educ': {'High school or less': 0, 'Some college': 1, 'College': 2},
            'ppage': {'25-': 0, '26-34': 1, '35-49': 2, '50-64': 3, '65+': 4}
        }

        # columns with string answers
        self._cols_with_string_answers = list(self._string_int_mapping.keys())

    def categorize_age(self, age_array):
        """
        This function is to categorize the ages of the respondants into different categories: '25-','26-34','35-49','50-64','65+'.
        """

        ages = age_array.astype(str)
        age_categories = [None]*ages.shape[0]
        for i in range(ages.shape[0]):
            if(int(ages[i])<=25):
                age_categories[i]='25-'
            elif(int(ages[i])>=26 and int(ages[i])<=34):
                age_categories[i]='26-34'
            elif(int(ages[i])>=35 and int(ages[i])<=49):
                age_categories[i]='35-49'
            elif(int(ages[i])>=50 and int(ages[i])<=64):
                age_categories[i]='50-64'
            else:
                age_categories[i]='65+'

        return np.array(age_categories)

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

        dtype = np.float32 if self._convert_to_float else object
        return_table = np.zeros((self._dataframe.shape[0], len(question_list)), dtype=dtype)

        for i, question in enumerate(question_list):
            # get column index, throw exception if question is not valid
            try:
                answers = self._dataframe[question]
            except KeyError:
                raise KeyError("{} is not a valid question in the poll".format(question))

            # categorize columns that need to be categorized
            if question == 'ppage':
                answers = self.categorize_age(answers)

            # get the data from the table and add it to the result
            if question in self._cols_with_string_answers and self._convert_to_float:
                return_table[:, i] = self._answers_as_numbers(answers, question)
            else:
                return_table[:, i] = answers

        # remove all rows with a nan value in them
        if self._remove_nan:
            if self._convert_to_float:
                return_table = return_table[~np.isnan(return_table).any(axis=1)]
            else:
                print("Warning: cannot remove nans from data which are not all floats")
        return return_table, question_list

    def all_data_except(self, question_list_to_remove=None):
        """
        Get all the responses for the specified poll questions.
        Args:
            question_list_to_remove (list(str)):
                A list of questions to 'NOT' get the response data for, defaults to
                zero questions.
        Returns (np.ndarray(N_answers, N_questions)):
            All of the answers for the questions in the poll except the ones specified in the argument.
        """

        if question_list_to_remove is None:
            question_list_to_remove = []

        final_questions = [x for x in self._list_all_question if x not in question_list_to_remove]
        return_table, questions = self.all_data(final_questions)

        return return_table, questions

    def questions(self):
        """
        Get a list of all the questions in the poll.

        Returns (list(str)):
            A list of the numbers for each question in the poll.
        """
        return list(self._dataframe.columns)

    def _answers_as_numbers(self, responses, column_name):
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
        responses_as_numbers = np.zeros(len(responses))
        for i, string_response in enumerate(responses):
            responses_as_numbers[i] = mapping_dict[string_response]
        return responses_as_numbers

    def __getattr__(self, column_name):
        """ Get the answers to each poll question as lists. """
        cols = self._dataframe.columns

        # get the answers to the question as int
        question_answers = self._dataframe[column_name]
        if column_name in self._cols_with_string_answers:
            if(self._convert_to_float):
                result = self._answers_as_numbers(question_answers, column_name)
            else:
                result = np.array(question_answers)
        elif column_name in cols:
            result = np.array(question_answers)
        else:
            raise RuntimeError("Cannot find column for {} in the poll data".format(column_name))

        # remove nan values
        if self._remove_nan:
            result = result[np.logical_not(np.isnan(result))]

        return result

