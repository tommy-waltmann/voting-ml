import data


def main():
    # make a data proxy object
    # it will include all the nan values in the table in all the data it returns
    poll_data = data.PollDataProxy()

    # access answers to questions as numpy arrays
    print(poll_data.Q1)
    print(poll_data.Q22)  # this questions is almost all nan's
    print(poll_data.Q28_6)
    print(poll_data.RespId)

    # for questions whose answers are strings, it returns ints corresponding to
    # those strings. The mappings between string answers and ints are in data.py
    print(poll_data.educ)
    print(poll_data.voter_category)

    # get list of all question titles
    questions = poll_data.questions()
    print(questions)

    # get all the response data at once (shape is (N_responses, N_questions))
    all_answers = poll_data.all_data()
    print(all_answers.shape)

    # get all the response data at once (shape is (N_responses, N_questions_in_list))
    question_set_answers = poll_data.all_data(["Q1", "Q27_6", "Q14", "educ"])
    print(question_set_answers.shape)

    # make a poll data object which will exclude nan values in the data it returns
    poll_data = data.PollDataProxy(remove_nan=True)

    print(poll_data.Q22)  # now the data has no nan values in it

    # removing nans from all_data involves removing all rows which have a nan
    # answer in them
    question_set_answers = poll_data.all_data(["Q2_1", "Q18_5", "Q22", "voter_category"])
    print(question_set_answers.shape)


if __name__ == "__main__":
    main()
