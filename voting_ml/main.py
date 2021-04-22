import numpy as np
import sklearn
import subprocess
from sklearn import model_selection, tree

import data
import feature_selection
import model_sel

import matplotlib.pyplot as plt
import seaborn as sns

def main():

    #parameter space
    list_test_size = [0.1,0.15,0.2] # decide this
    list_ftsel_method = ['chi2','mutlinfo','pca','dt']
    list_num_features = [] # decide this
    list_Kfold = [3,5]
    list_corr_threshold = [0.5] # decide this
    
    #output dictrionary list
    list_output_dict = []

    # output directory path
    outdir = "../results/run1/"
    
    #splitting data and weights into train, test (refer to optimal_params.py)
    '''refer to optimal_params.py. Functions from this python scripts are transferred here. (get_bad_questions() and separate_weights().)'''
    for ts in list_test_size:
        X_train, X_test, y_train, y_test = 
        X_train, weights_train = 
        X_test, weights_test = 
        print("Number of Training Samples:", len(X_train))
        print("Number of Testing Samples:", len(X_test))

        data_dict = {
        }
        weights_dict = {
        }
        
        for meth in list_ftsel_method:
            '''Create class objects of the current selection method'''
            for thres in list_corr_threshold:
                data_sel_dict, sel_questions = {}, []
                if(meth=='chi2'):
                    ftsel_obj = 
                    data_sel_dict, sel_questions = 
                elif(meth=='mutlinfo'):
                    ftsel_obj =
            	    data_sel_dict, sel_questions =
                elif(meth=='pca'):
                    ftsel_obj =
            	    data_sel_dict, sel_questions =
                elif(meth=='dt'):
                    ftsel_obj =
            	    data_sel_dict, sel_questions =
                for num in list_num_features:
                
                    for K in Kfold:
                        '''Here create a class onject of "model_sel" and output all the best parameters and values into "list_output_dict". Then, can create a .csv file to list all the models and accuracies.'''
                        
    '''Once all the models are run, select the model with best test accuracy and return the output dict for that model.'''

    best_model_dict = list_output_dict[best_index]

    print("The best model parameters:")
    print(best_model_dict)
    
                    
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
