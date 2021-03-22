import data
import graphviz
import sklearn
from sklearn import tree


def ordered_difference(list1, list2):
    out_list = []
    for elt in list1:
        if elt not in list2:
            out_list.append(elt)
    return out_list


def get_feature_questions(poll_data):
    """ Return a list of questions whose answers will be used as features. """
    """
    # question set which onyl excludes q's that have a lot of nan values
    q29 = ["Q29_{}".format(i) for i in range(1, 11)]
    nonfeature_questions = ["RespId", "weight", "ppage", "Q22", "Q31", "Q32", "Q33"] + q29
    feature_questions = ordered_difference(poll_data.questions(), nonfeature_questions)
    """

    # first guess as to which features might be important
    q2 = ["Q2_{}".format(i) for i in range(1, 11)]
    q3 = ["Q3_{}".format(i) for i in range(1, 7)]
    q4 = ["Q4_{}".format(i) for i in range(1, 7)]
    q5 = ["Q5"]
    q6 = ["Q6"]
    q7 = ["Q7"]
    q8 = ["Q8_{}".format(i) for i in range(1, 10)]
    q9 = ["Q9_{}".format(i) for i in range(1, 5)]
    q10 = ["Q10_{}".format(i) for i in range(1, 5)]
    q11 = ["Q11_{}".format(i) for i in range(1, 7)]
    q14 = ["Q14"]
    q15 = ["Q15"]
    q16 = ["Q16"]
    q17 = ["Q17_{}".format(i) for i in range(1, 5)]
    q18 = ["Q18_{}".format(i) for i in range(1, 11)]
    q19 = ["Q19_{}".format(i) for i in range(1, 11)]
    q20 = ["Q20"]
    q21 = ["Q21"]
    q22 = ["Q22"]
    q23 = ["Q23"]
    q24 = ["Q24"]
    q25 = ["Q25"]
    q26 = ["Q26"]
    q27 = ["Q27_{}".format(i) for i in range(1, 7)]
    q28 = ["Q28_{}".format(i) for i in range(1, 9)]
    q29 = ["Q29_{}".format(i) for i in range(1, 11)]

    feature_questions = q2 +


    return feature_questions


def main():
    poll_data = data.PollDataProxy(remove_nan=True)

    feature_questions = get_feature_questions(poll_data)
    responses = poll_data.all_data(feature_questions)
    print("Number of samples: {}".format(len(responses)))
    print("Number of features: {}".format(len(responses[0]) - 1))

    outputs = responses[:, -1]
    inputs = responses[:, :-1]
    train_x = inputs[:4500]
    train_y = outputs[:4500]
    test_x = inputs[4500:]
    test_y = outputs[4500:]

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_x, train_y)
    predicted_y = classifier.predict(test_x)
    perc_correct = sklearn.metrics.accuracy_score(predicted_y, test_y)
    print("Percent correct: {}".format(perc_correct))

    feature_names = ordered_difference(feature_questions, ["voter_category"])
    class_names = ["rarely/never", "sporadic", "always"]
    graph_data = tree.export_graphviz(classifier,
                                      out_file="graph.dot",
                                      feature_names=feature_names,
                                      class_names=class_names,
                                      filled=True,
                                      rounded=True,
                                      special_characters=True)



if __name__ == "__main__":
    main()
