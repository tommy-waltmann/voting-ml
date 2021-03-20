import data
import sklearn


def main():
    poll_data = data.PollDataProxy()
    features = poll_data.all_data()

    classifier = sklearn.tree.DecisionTreeClassifier()
    classifier.fit()


if __name__ == "__main__":
    main()
