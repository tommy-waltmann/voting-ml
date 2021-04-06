import data
import numpy as np
import sklearn
from sklearn import model_selection, tree

def main():
    # grab data
    poll_data = data.PollDataProxy(remove_nan=True, convert_to_int=True)
    all_data, all_questions = poll_data.all_data(['Q16', 'Q20', 'voter_category'])
    np.random.shuffle(all_data)
    print("Number of Samples:", len(all_data))

    # set up train/validate and test datasets
    num_train_samples = 5000
    inputs = all_data[:, :-1]
    outputs = all_data[:, -1]
    train_x = inputs[:num_train_samples]
    train_y = outputs[:num_train_samples]
    test_x = inputs[num_train_samples:]
    test_y = outputs[num_train_samples:]

    # dictionary which sets the parameters
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
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    print("Best parameters:\n{}".format(best_params))

    # train the model with the best parameters, and report test/train accuracy
    clf = tree.DecisionTreeClassifier(**best_params)
    clf.fit(test_x, test_y)
    pred_test_y = clf.predict(test_x)
    test_acc = sklearn.metrics.accuracy_score(pred_test_y, test_y)
    print("Test Accuracy: {}".format(test_acc))
    pred_train_y = clf.predict(train_x)
    train_acc = sklearn.metrics.accuracy_score(pred_train_y, train_y)
    print("Train Accuracy: {}".format(train_acc))


if __name__ == "__main__":
    main()
