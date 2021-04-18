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
from sklearn.feature_selection import chi2,mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class FeatureSelection:
    ''' This class provides several methods to select features from the given data.
    Main two methods are chi2 test and mutual information method.
    This class also generates some correrlation graphs between different features
    '''

    def __init__(self, necess_que_file, unnecess_que_file, bool_necess_que=False, run_name="test"):
        
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

    def get_ftsel_original_data(self):

        return self._ftsel_data
    
    def split_data(self, X, y, input_test_size=0, input_random_state=None):

        data_dict = {}
        if(input_test_size==0):
            data_dict["X_train"] = X
            data_dict["X_test"] = None
            data_dict["y_train"] = y
            data_dict["y_test"] = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=input_test_size, random_state=input_random_state)
            data_dict["X_train"] = X_train
            data_dict["X_test"] = X_test
            data_dict["y_train"] = y_train
            data_dict["y_test"] = y_test

        return data_dict
            
    def ftsel_chi2(self, data_dict, KBest = 'all'):
        
        X_train = data_dict["X_train"].astype(str)
        X_test = data_dict["X_test"].astype(str)
        y_train = data_dict["y_train"].astype(str)
        y_test = data_dict["y_test"].astype(str)
        
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        
        # prepare input data
        X_train_enc, X_test_enc, oe = self.prepare_inputs(X_train, X_test)
        
        # prepare output data
        y_train_enc, y_test_enc, le = self.prepare_targets(y_train, y_test)
        
        # feature selection
        X_train_fs, X_test_fs, fs = self.ftsel_KBest(X_train_enc, y_train_enc, X_test_enc, KBest, score_fn=chi2)
        
        # what are scores for the features
        ft_num = np.arange(len(fs.scores_)).reshape(len(fs.scores_),1)
        ft_num = ft_num.astype(int)
        fs_scores = fs.scores_.reshape(len(fs.scores_),1)
        fs_p_values = fs.pvalues_.reshape(len(fs.scores_),1)
        fs_ft_scores = np.concatenate((ft_num,fs_scores),axis=1)
        fs_ft_sc_pv = np.concatenate((fs_ft_scores,fs_p_values),axis=1)
        fs_ft_scores_sort = np.sort(fs_ft_sc_pv.view('i8,i8,i8'), order=['f1'], axis=0).view(np.float)[::-1]
        for i in range(1,len(fs.scores_)+1,1):
            print('Chi2 - Feature %d - %d: %f and p-value: %f' % (i,fs_ft_scores_sort[i-1][0], fs_ft_scores_sort[i-1][1], fs_ft_scores_sort[i-1][2]))
        
        #Creating directory for the output
        if(not os.path.isdir("../output/"+self._run_name)):
            os.mkdir("../output/"+self._run_name)
        
        # plot the scores
        plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
        image = "../output/"+self._run_name+"/"+"chi2_scores_versus_features_barplot.png"
        plt.savefig(image)
        plt.clf()
        
        # Plot scores
        plt.plot(ft_num,fs_ft_scores_sort[:,1], label = "score versus feature rank")
        plt.xlabel('feature rank')
        plt.ylabel('score')
        plt.legend()
        image_name = "../output/"+self._run_name+"/"+"chi2_score_versus_feature_rank_ftsel_data.png"
        plt.savefig(image_name)
        plt.clf()

        #Returning selected features and labels
        ftsel_X_train = np.empty( (N_train,(KBest)), dtype=object)
        ftsel_X_test = np.empty( (N_test,(KBest)), dtype=object)
        
        ftsel_questions = [None] * KBest
        
        for k in range(KBest):
            ftsel_X_train[:,k] = X_train[:,int(fs_ft_scores_sort[k][0])]
            ftsel_X_test[:,k] = X_test[:,int(fs_ft_scores_sort[k][0])]
            ftsel_questions[k] = self._ftsel_quelist[int(fs_ft_scores_sort[k][0])]
            
        ftsel_data_dict = {
            'X_train': ftsel_X_train,
            'X_test': ftsel_X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        str_questions = ",".join(ftsel_questions)
        o_ftsel_que_file = open("../output/"+self._run_name+"/"+"chi2_ftsel_questions_list.txt","w")
        o_ftsel_que_file.write(str_questions)
        o_ftsel_que_file.close()
        
        return ftsel_data_dict, ftsel_questions
    
    def ftsel_mutlinfo(self, data_dict, KBest = 'all'):
 
        X_train = data_dict["X_train"].astype(str)
        X_test = data_dict["X_test"].astype(str)
        y_train = data_dict["y_train"].astype(str)
        y_test = data_dict["y_test"].astype(str)

        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        
        # prepare input data
        X_train_enc, X_test_enc, oe = self.prepare_inputs(X_train, X_test)
        
        # prepare output data
        y_train_enc, y_test_enc, le = self.prepare_targets(y_train, y_test)
        
        # feature selection
        X_train_fs, X_test_fs, fs = self.ftsel_KBest(X_train_enc, y_train_enc, X_test_enc, KBest, score_fn=mutual_info_classif)
        
        # what are scores for the features
        ft_num = np.arange(len(fs.scores_)).reshape(len(fs.scores_),1)
        ft_num = ft_num.astype(int)
        fs_scores = fs.scores_.reshape(len(fs.scores_),1)
        fs_ft_scores = np.concatenate((ft_num,fs_scores),axis=1)
        fs_ft_scores_sort = np.sort(fs_ft_scores.view('i8,i8'), order=['f1'], axis=0).view(np.float)[::-1]
        for i in range(1,len(fs.scores_)+1,1):
            print('Feature %d - %d: %f ' % (i,fs_ft_scores_sort[i-1][0], fs_ft_scores_sort[i-1][1]))

        #Creating directory for the output
        if(not os.path.isdir("../output/"+self._run_name)):
            os.mkdir("../output/"+self._run_name)
        
        # plot the scores
        plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
        image = "../output/"+self._run_name+"/"+"mutlinfo_scores_versus_features_barplot.png"
        plt.savefig(image)
        plt.clf()

        # Plot scores
        plt.plot(ft_num,fs_ft_scores_sort[:,1], label = "score versus feature rank")
        plt.xlabel('feature rank')
        plt.ylabel('score')
        plt.legend()
        image_name = "../output/"+self._run_name+"/"+"mutlinfo_score_versus_feature_rank_ftsel_data.png"
        plt.savefig(image_name)
        plt.clf()

        #Returning selected features and labels
        ftsel_X_train = np.empty( (N_train,(KBest)), dtype=object)
        ftsel_X_test = np.empty( (N_test,(KBest)), dtype=object)

        ftsel_questions = [None] * KBest

        for k in range(KBest):
            ftsel_X_train[:,k] = X_train[:,int(fs_ft_scores_sort[k][0])]
            ftsel_X_test[:,k] = X_test[:,int(fs_ft_scores_sort[k][0])]
            ftsel_questions[k] = self._ftsel_quelist[int(fs_ft_scores_sort[k][0])]

        ftsel_data_dict = {
            'X_train': ftsel_X_train,
            'X_test': ftsel_X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        str_questions = ",".join(ftsel_questions)
        o_ftsel_que_file = open("../output/"+self._run_name+"/"+"mutlinfo_ftsel_questions_list.txt","w")
        o_ftsel_que_file.write(str_questions)
        o_ftsel_que_file.close()

        return ftsel_data_dict, ftsel_questions

    def ftsel_pca(self, data_dict, KBest=20):

        X_train = data_dict["X_train"].astype(str)
        X_test = data_dict["X_test"].astype(str)
        y_train = data_dict["y_train"].astype(str)
        y_test = data_dict["y_test"].astype(str)

        # prepare input data
        print("X_train",X_train)
        X_train_enc, X_test_enc, oe = self.prepare_inputs(X_train, X_test)

        # prepare output data
        y_train_enc, y_test_enc, le = self.prepare_targets(y_train, y_test)
    
        pca = PCA()
        pca.fit(X_train_enc)
        transform_matrix = pca.components_.T
        eigen_vals = pca.explained_variance_

        #Creating directory for the output
        if(not os.path.isdir("../output/"+self._run_name)):
            os.mkdir("../output/"+self._run_name)
        
        print("Eigen values for first 20 principal components: ",eigen_vals[0:20])
        rank_eigenval = np.arange(1,X_train_enc.shape[1]+1).reshape((X_train_enc.shape[1]))
        plt.plot(rank_eigenval,eigen_vals, label = "eigen-values")
        plt.xlabel('Rank of the eigenvalue')
        plt.ylabel('Eigen Value')
        plt.yscale('log')
        plt.title("PCA - eigenvalues")
        plt.legend()
        image_name = '../output/'+self._run_name+'/eigenvalues_vs_rank.png'
        plt.savefig(image_name)
        plt.clf()

        total_var = np.zeros((eigen_vals.shape[0]))
        total_sum = np.sum(eigen_vals)
        for i in range(eigen_vals.shape[0]):
            total_var[i] = np.sum(eigen_vals[0:i+1])*100/total_sum

        num_95perVar = None
        num_99perVar = None

        for j in range(eigen_vals.shape[0]):
            if(total_var[j]>=95):
                num_95perVar = j+1
                break

        for k in range(eigen_vals.shape[0]):
            if(total_var[k]>=99):
                num_99perVar = k+1
                break

        print("Number of principal components needed to represent 95% of the total variance: ", num_95perVar)
        print("Number of principal components needed to represent 99% of the total variance: ", num_99perVar)

        pca_data = data
        pca_X_train = np.dot(X_train_enc,transform_matrix)
        pca_X_test = np.dot(X_test_enc,transform_matrix)

        pca_data_dict = {
            'X_train': pca_X_train[:,0:KBest],
            'X_test': pca_X_test[:,0:KBest],
            'y_train': y_train,
            'y_test': y_test
        }

        pca_components_dict = {
            'transform_matrix': transform_matrix,
            'eigen_vals': eigen_vals,
            'KBest': KBest,
            'total_var': total_var
        }
        
        return pca_data_dict, pca_components_dict
    
    def ft_corr(self, X_train, questions):
        if(questions==None):
            fts = X_train.shape[1]
            questions_int = list(map(str, list(range(1,fts+1,1))))
            questions = ["ft_"+x for x in questions_int]
        KBest = len(questions)
        df = pd.DataFrame(X_train, columns = questions)
        le=LabelEncoder()
        for column in df.columns:
            df[column] = le.fit_transform(df[column])
        df_corr = df.corr(method='pearson')
        
        plt.figure(figsize=(15,15))
        sns.heatmap(df_corr,linewidths=.1,cmap="YlGnBu", annot=True)
        plt.yticks(rotation=0)
        #plt.show()
        image_name = "../output/"+self._run_name+"/"+"ft_corr_KBest_"+str(KBest)+".png"
        plt.savefig(image_name)
        plt.clf()
        return df_corr
        
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
        return X_train_enc, X_test_enc, oe

    def prepare_targets(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc, le

    def ftsel_KBest(self, X_train, y_train, X_test, K, score_fn=chi2):
        fs = SelectKBest(score_func=score_fn, k=K)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs
    
    def def Nmaxelements(self, list1, N):
        final_list = []
        for i in range(0, N): 
            max1 = 0 
            for j in range(len(list1)):
                if list1[j] > max1:
                    max1 = list1[j];

            list1.remove(max1);
            final_list.append(max1)

        return final_list
        
    def ftsel_decision_tree_method(self, data_dict, num_features = 20):

        X_train = data_dict["X_train"].astype(str)
        X_test = data_dict["X_test"].astype(str)
        y_train = data_dict["y_train"].astype(str)
        y_test = data_dict["y_test"].astype(str)

        # prepare input data
        X_train_enc, X_test_enc, oe = self.prepare_inputs(X_train, X_test)

        # prepare output data
        y_train_enc, y_test_enc, le = self.prepare_targets(y_train, y_test)

        model = DecisionTreeClassifier()
        model.fit(X_train_enc, y_train_enc)

        importance = model.feature_importances_

        best_fts_scores = Nmaxelements(list(importance), num_features)
        best_fts = []


        for score in best_fts_scores:
            ind = np.where(importance == score)
            best_fts.append(df.columns[ind][0])

        plt.bar(best_fts, best_fts_scores)
        plt.xticks(rotation = 90)
        plt.xlabel('Features')
        plt.ylabel('Score')
        plt.title("Scores of the best 20 features")
        image_name = '../output/'+self._run_name+'/scores-from-decision-tree-method.png'
        plt.savefig(image_name)
        plt.show()

        if not os.path.exists("../output/"+self._run_name+"/"):
            os.makedirs("../output/"+self._run_name+"/")

        str_questions = ",".join(best_fts)
        o_ftsel_que_file = open("../output/"+self._run_name+"/"+"Decision_tree_ftsel_questions_list.txt","w")
        o_ftsel_que_file.write(str_questions)
        o_ftsel_que_file.close()

        return ftsel_data_dict, best_fts   
