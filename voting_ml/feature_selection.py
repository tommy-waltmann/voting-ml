# Adapted from https://machinelearningmastery.com/feature-selection-with-categorical-data/#:~:text=The%20two%20most%20commonly%20used,and%20the%20mutual%20information%20statistic.  
import numpy as np
import pandas as pd
import os
import data
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

class FeatureSelection:
    ''' This class provides several methods to select features from the given data.
    Main two methods are chi2 test and mutual information method.
    This class also generates some correrlation graphs between different features
    '''

    def __init__(self, necess_que_file, unnecess_que_file, bool_necess_que=False, run_name="test"):
        
        self._dataframe = pd.read_csv(
            "../extern/fivethirtyeight_data/non-voters/nonvoters_data.csv"
        )
        
        self._bool_necess_que = bool_necess_que
        self._run_name = run_name
        self._list_unnecess_que = None
        self._list_necess_que = None
        self._ftsel_data = None
        self._ftsel_quelist = None
        self._list_unnecess_que = []
        self._list_necess_que = []
        
        if(not self._bool_necess_que):
            self._unnecess_que_file = open(unnecess_que_file,"r")
            nonempty_lines_unnecess = [line.strip("\n") for line in self._unnecess_que_file if not line.isspace()]
            if(len(nonempty_lines_unnecess)!=1):
                raise Exception("The file containing the unnecessary questions has more than one lines. All question names should be written in one line separated by commas.")
            else:
                try:
                    self._list_unnecess_que = nonempty_lines_unnecess[0].split(",")
                except KeyError:
                    raise KeyError("The line containing unnecessary questions cannot be split to get a list of questions. Please make sure the questions in the file are separated by comma.")
        else:
            self._necess_que_file = open(necess_que_file,"r")

            nonempty_lines_necess = [line.strip("\n") for line in self._necess_que_file if not line.isspace()]
            if(len(nonempty_lines_necess)!=1):
                raise Exception("The file containing the necessary questions has more than one lines. All question names should be written in one line separated by commas.")
            else:
                try:
                    self._list_necess_que = nonempty_lines_necess[0].split(",")
                except KeyError:
                    raise KeyError("The line containing necessary questions cannot be split to get a list of questions. Please make sure the questions in the file are separated by comma.")
        
        self._poll_data = data.PollDataProxy(remove_nan=False, convert_to_int=False)

        if(self._bool_necess_que):
            self._ftsel_data, self._ftsel_quelist = self._poll_data.all_data(self._list_necess_que)
        else:
            self._ftsel_data, self._ftsel_quelist = self._poll_data.all_data_except(self._list_unnecess_que)

        print("self._ftsel_data",self._ftsel_data.shape)
            
    def ftsel_chi2(self, KBest = 'all', label=None):
        
        X, y = self.separate_ft_label(self._ftsel_data)
        N = X.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        
        # prepare input data
        X_train_enc, X_test_enc = self.prepare_inputs(X_train, X_test)
        
        # prepare output data
        y_train_enc, y_test_enc = self.prepare_targets(y_train, y_test)
        
        # feature selection
        X_train_fs, X_test_fs, fs = self.ftsel_KBest(X_train_enc, y_train_enc, X_test_enc, KBest)
        
        # what are scores for the features
        ft_num = np.arange(len(fs.scores_)).reshape(len(fs.scores_),1)
        ft_num = ft_num.astype(int)
        fs_scores = fs.scores_.reshape(len(fs.scores_),1)
        fs_ft_scores = np.concatenate((ft_num,fs_scores),axis=1)
        fs_ft_scores_sort = np.sort(fs_ft_scores.view('i8,i8'), order=['f1'], axis=0).view(np.float)[::-1]
        for i in range(1,len(fs.scores_)+1,1):
            print('Feature %d - %d: %f' % (i,fs_ft_scores_sort[i-1][0], fs_ft_scores_sort[i-1][1]))

        #Creating directory for the output
        if(not os.path.isdir("../output/"+self._run_name)):
            os.mkdir("../output/"+self._run_name)
        
        # plot the scores
        plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
        image = "../output/"+self._run_name+"/"+"scores_versus_features_barplot.png"
        plt.savefig(image)
        plt.clf()

        # Plot scores
        plt.plot(ft_num,fs_ft_scores_sort[:,1], label = "score versus feature rank")
        plt.xlabel('feature rank')
        plt.ylabel('score')
        plt.legend()
        image_name = "../output/"+self._run_name+"/"+"score_versus_feature_rank_ftsel_data.png"
        plt.savefig(image_name)
        plt.clf()

        #Returning selected features and labels
        data_selft = np.empty(N*(KBest+1), dtype=object)
        data_selft = data_selft.reshape((N,KBest+1))

        selft_question = [None] * KBest
        
        for k in range(KBest):
            data_selft[:,k] = self._ftsel_data[:,int(fs_ft_scores_sort[k][0])]
            selft_question[k] = self._ftsel_quelist[int(fs_ft_scores_sort[k][0])]
            
        data_selft[:,KBest] = y

        str_questions = ",".join(selft_question)
        o_ftsel_que_file = open("../output/"+self._run_name+"/"+"ftsel_questions_list.txt","w")
        o_ftsel_que_file.write(str_questions)
        o_ftsel_que_file.close()

        return data_selft, selft_question        

    def separate_ft_label(self, dataset, label=None):
        X = None
        y = None
        if(label==None):
            X = dataset[:, :-1]
            y = dataset[:,-1]
        else:
            X = np.delete(dataset, label, 1)
            y = dataset[:,label]

        X = X.astype(str)
        return X, y
            
    def prepare_inputs(self, X_train, X_test):
        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc

    def prepare_targets(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    def ftsel_KBest(self, X_train, y_train, X_test, K):
        fs = SelectKBest(score_func=chi2, k=K)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs
    
    

        
