import numpy as np
import sklearn
import subprocess
from sklearn import model_selection, tree

import data
import feature_selection


def ordered_difference(list1, list2):
    out_list = []
    for elt in list1:
        if elt not in list2:
            out_list.append(elt)
    return out_list


def get_bad_questions():
    f = open("../extern/manage_data/list_unnecessary_columns.txt", 'r')
    bad_questions = f.readline().split(',')
    bad_questions[-1] = bad_questions[-1][:-1]  # chop the \n off the end
    bad_questions.remove('weight')  # need weight for training
    return bad_questions


def separate_weights(X_train, column_names):
    """
    Removes the column containing weights from X_train, and returns it as
    a separate array.
    """
    weight_column_idx = column_names.index('weight')
    weights = X_train[:, weight_column_idx]
    new_X_train = np.delete(X_train, weight_column_idx, axis=1)
    return new_X_train, weights


def main():
    # grab and split data
    poll_data = data.PollDataProxy(remove_nan=True, convert_to_float=True)
    all_data, all_data_questions = poll_data.all_data_except(get_bad_questions())
    X = all_data[:, :-1]
    y = all_data[:, -1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.1,
                                                                        shuffle=True)
    X_train, weights_train = separate_weights(X_train, all_data_questions[:-1])
    X_test, weights_test = separate_weights(X_test, all_data_questions[:-1])
    print("Number of Training Samples:", len(X_train))
    print("Number of Testing Samples:", len(X_test))

    # create data dictionary and cast values to ints
    data_dict = {
        'X_train': X_train.astype(np.int32),
        'X_test': X_test.astype(np.int32),
        'y_train': y_train.astype(np.int32),
        'y_test': y_test.astype(np.int32)
    }

    # get best features
    chi2_ftsel = feature_selection.FeatureSelection(
        necess_que_file="../extern/manage_data/list_all_questions.txt",
        unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
        bool_necess_que=False,
        run_name="test_chi2"
    )
    data_ftsel_dict, ftsel_questions = chi2_ftsel.ftsel_chi2(data_dict, KBest=20)
    ftsel_X_train = data_ftsel_dict['X_train']
    ftsel_X_test = data_ftsel_dict['X_test']
    ftsel_y_train = data_ftsel_dict['y_train']
    ftsel_y_test = data_ftsel_dict['y_test']

    # dictionary for hyperparameter search
    param_space = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_leaf_nodes': [2, 5, 10, 20, 30],
    }

    # determine best parameters
    dt_clf = tree.DecisionTreeClassifier()
    grid_search = model_selection.GridSearchCV(dt_clf, param_space, cv=5,
                                               scoring='accuracy', verbose=1)
    grid_search.fit(ftsel_X_train, ftsel_y_train, sample_weight=weights_train)
    best_params = grid_search.best_params_
    print("Best parameters:\n{}".format(best_params))

    # train the model with the best parameters, and report test/train accuracy
    clf = tree.DecisionTreeClassifier(**best_params)
    clf.fit(ftsel_X_train, ftsel_y_train, sample_weight=weights_train)
    pred_y_test = clf.predict(ftsel_X_test)
    test_acc = sklearn.metrics.accuracy_score(pred_y_test, ftsel_y_test)
    print("Test Accuracy: {}".format(test_acc))
    pred_y_train = clf.predict(ftsel_X_train)
    train_acc = sklearn.metrics.accuracy_score(pred_y_train, ftsel_y_train)
    print("Train Accuracy: {}".format(train_acc))

    # write the graph data to a dot file
    class_names = ['rarely/never', 'sporadic', 'always']
    graph_data = tree.export_graphviz(clf,
                                      out_file="graph.dot",
                                      feature_names=ftsel_questions,
                                      class_names=class_names,
                                      filled=True,
                                      rounded=True,
                                      special_characters=True)
    # write the .dot file to a png
    command = "dot -Tpng graph.dot -o graph1.png"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()


if __name__ == "__main__":
    main()
