import pandas as pd


class PollDataProxy:
    """
    This class acts as a proxy to all of the data in the poll used for this
    assignment. It has methods that access the data using a pandas dataframe
    object, and ideally all the pandas in this project will appear in this file
    and only in this file.
    """

    def __init__(self):
        self._dataframe = pd.read_csv(
            "../extern/fivethirtyeight_data/non-voters/nonvoters_data.csv"
        )

    def __getattr__(self, column_name):
        """ Get the answers to each poll question as lists. """
        cols = self._dataframe.columns
        if column_name in cols:
            return list(self._dataframe[column_name])
        else:
            raise RuntimeError(
                "Cannot find column for {} in the poll data".format(column_name))
