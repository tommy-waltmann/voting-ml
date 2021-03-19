import data


def main():
    poll_data = data.PollDataProxy()

    # access answers to questions as numpy arrays
    print(poll_data.Q1)
    print(poll_data.Q28_6)
    print(poll_data.RespId)

    # for questions whose answers are strings, it returns ints corresponding to
    # those strings. The mappings between string answers and ints are in data.py
    print(poll_data.educ)
    print(poll_data.voter_category)


if __name__ == "__main__":
    main()
