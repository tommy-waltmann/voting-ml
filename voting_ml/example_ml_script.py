import data
import sklearn


def main():
    poll_data = data.PollDataProxy()
    print(poll_data.Q1)
    print(poll_data.Q28_6)
    print(poll_data.educ)
    print(poll_data.RespId)


if __name__ == "__main__":
    main()
