import numpy as np
import sklearn
import subprocess
from sklearn import model_selection, tree

import data
import feature_selection

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import os

class model_sel:
    '''This class defines all the basic parameters of a model. Creating such classes for different sets
    of parameters will give accuracies from using different sets. Implementation in main.py.
    '''
    def __init__(self,test_size,run_num,ftsel_method,param_space,Kfold,num_features,threshold,data_ftsel_dict,weights_dict,questions,outdir):

        '''
        Here,
        test_size : is a fraction of the total data that will be stored as test data. e.g. 0.1 or 0.2, etc.
        ftsel_method : feature selection method one of ['chi2','mutlinfo','pca','dt']
        Kfold : K value for Kfold cross-validation in decision tree classifier grid search.
        data_ftsel_dict : dictionary of train and test data ranked after using one of the above feature selection methods. (best features not yet selected. only ranked.) So, in main.py, you have to use the feature selection method only to rank the features and not to select them and pass that dictionary to this class. According to num_features in this class, the class will select those many features.
        num_features : num_features to be selected from the given ranked train and test data.
        weights_dict : sample weights for train and test data
        questions : ranked feature-names corresponding to the train and test data
        outdir : output directory path where all the plots and output files will be stored.
        '''

        self._test_size = test_size
        self._run_num = run_num
        self._ftsel_method = ftsel_method
        self._Kfold = Kfold
        self._num_features = num_features
        self._corr_threshold = threshold
        self._X_train = data_ftsel_dict['X_train']
        self._X_test = data_ftsel_dict['X_test']
        self._y_train = data_ftsel_dict['y_train']
        self._y_test = data_ftsel_dict['y_test']
        self._questions = questions[0:self._num_features]
        self._weights_train = weights_dict['weights_train']
        self._weights_test = weights_dict['weights_test']

        if(self._ftsel_method!='pca'):
            # prepare input data
            self._X_train_enc, self._X_test_enc, self._oe = self.prepare_inputs(self._X_train, self._X_test)
            # prepare output data
            self._y_train_enc, self._y_test_enc, self._le = self.prepare_targets(self._y_train, self._y_test)
        else:
            self._X_train_enc = self._X_train
            self._X_test_enc = self._X_test
            self._y_train_enc = self._y_train
            self._y_test_enc = self._y_test
        self._param_space = param_space
        '''
        self._param_space = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 3, 4, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
            'max_leaf_nodes': [2, 4, 6, 8, 10, 12, 15],
        }
        '''
        self._outdir = outdir
        self._run_name = "ts{0}_run{1}_{2}_Nfts{3}_Kfold{4}_thres{5}".format(self._test_size,self._run_num,self._ftsel_method,self._num_features,self._Kfold,self._corr_threshold)
        if(not os.path.isdir(self._outdir+self._run_name)):
            os.mkdir(self._outdir+self._run_name)

    def select_model(self):
        # determine best parameters
        self._dt_clf = tree.DecisionTreeClassifier()
        grid_search = model_selection.GridSearchCV(self._dt_clf, self._param_space, cv=self._Kfold,
                                               scoring='accuracy', verbose=1)
        grid_search.fit(self._X_train_enc, self._y_train_enc, sample_weight=self._weights_train)
        self.best_params = grid_search.best_params_
        print("Best parameters:\n{}".format(self.best_params))
        # train the model with the best parameters, and report test/train accuracy
        self._clf = tree.DecisionTreeClassifier(**self.best_params)
        self._clf.fit(self._X_train_enc, self._y_train_enc, sample_weight=self._weights_train)

        self._train_acc = self._clf.score(self._X_train_enc,self._y_train_enc,self._weights_train)
        print("Train Accuracy: {}".format(self._train_acc))
        self._test_acc = self._clf.score(self._X_test_enc,self._y_test_enc,self._weights_test)
        print("Test Accuracy: {}".format(self._test_acc))

        # write the graph data to a dot file
        class_names = ['rarely/never', 'sporadic', 'always']
        graph_data = tree.export_graphviz(self._clf,
                                      out_file=self._outdir+self._run_name+"/graph.dot",
                                      feature_names=self._questions,
                                      class_names=class_names,
                                      filled=True,
                                      rounded=True,
                                      special_characters=True)
        # write the .dot file to a png
        command = "dot -Tpng "+self._outdir+self._run_name+"/graph.dot -o "+self._outdir+self._run_name+"/graph.png"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()

        self._model_sel_dict = {
            'test_size' : self._test_size,
            'run_num' : self._run_num,
            'ftsel_method' : self._ftsel_method,
            'Kfold' : self._Kfold,
            'num_features' : self._num_features,
            'corr_threshold' : self._corr_threshold,
            'best_features' : self._questions,
            'best_params' : self.best_params,
            'train_acc' : self._train_acc,
            'test_acc' : self._test_acc,
            'tree' : self._outdir+self._run_name+"/graph.dot"
        }

        return self._model_sel_dict

    def prepare_inputs(self, X_train, X_test):
        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc, oe

    def prepare_targets(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc, le

