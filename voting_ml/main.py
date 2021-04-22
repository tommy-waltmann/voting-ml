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
    list_num_features = [10,15,20] # decide this
    list_Kfold = [3,5]
    list_corr_threshold = [0.5,0.6,0.7] # decide this
    
    #output dictrionary list
    list_output_dict = []

    # output directory path
    outdir = "../results/run1/"
    
    #splitting data and weights into train, test (refer to optimal_params.py)
    poll_data = data.PollDataProxy(remove_nan=True, convert_to_float=True)
    
    acc = []
    
    '''refer to optimal_params.py. Functions from this python scripts are transferred here. (get_bad_questions() and separate_weights().)'''
    for ts in list_test_size:
        
        all_data, all_data_questions = poll_data.all_data_except(get_bad_questions())
        X = all_data[:, :-1]
        y = all_data[:, -1]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=ts,
                                                                        shuffle=True)
        X_train, weights_train = separate_weights(X_train, all_data_questions[:-1])
        X_test, weights_test = separate_weights(X_test, all_data_questions[:-1])

        print("Number of Training Samples:", len(X_train))
        print("Number of Testing Samples:", len(X_test))

        
        data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
        }
        weights_dict = { 
            'weights_train': weights_train,
            'weights_test': weights_test}
        
        for meth in list_ftsel_method:
            '''Create class objects of the current selection method'''
            for thres in list_corr_threshold:
                data_sel_dict, sel_questions = {}, []
                if(meth=='chi2'):
                   ftsel_obj = feature_selection.FeatureSelection(
                               necess_que_file="../extern/manage_data/list_all_questions.txt",
                               unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                               bool_necess_que=False,
                               run_name="test_chi2"
                               )
                   data_sel_dict, sel_questions = ftsel_obj.ftsel_chi2(data_dict, thres)
                elif(meth=='mutlinfo'):
                   ftsel_obj = feature_selection.FeatureSelection(
                               necess_que_file="../extern/manage_data/list_all_questions.txt",
                               unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                               bool_necess_que=False,
                               run_name="test_mutlinfo"
                               )
                   data_sel_dict, sel_questions = ftsel_obj.ftsel_mutlinfo(data_dict, thres)
                elif(meth=='pca'):
                   ftsel_obj = feature_selection.FeatureSelection(
                               necess_que_file="../extern/manage_data/list_all_questions.txt",
                               unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                               bool_necess_que=False,
                               run_name="test_pca"
                               )
                   data_sel_dict,_ = ftsel_obj.ftsel_pca(data_dict)
                   fts = data_sel_dict['X_train'].shape[1]
                   questions_int = list(map(str, list(range(1,fts+1,1))))
                   sel_questions = ["ft_"+x for x in questions_int]
                elif(meth=='dt'):
                   ftsel_obj = feature_selection.FeatureSelection(
                               necess_que_file="../extern/manage_data/list_all_questions.txt",
                               unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                               bool_necess_que=False,
                               run_name="test_dt"
                               )
                   data_sel_dict, sel_questions = ftsel_obj.ftsel_decision_tree_method(data_dict, thres)
                for num in list_num_features:
                
                    for K in list_Kfold:
                        '''Here create a class onject of "model_sel" and output all the best parameters and values into "list_output_dict". Then, can create a .csv file to list all the models and accuracies.'''
                        model_obj = model_sel.model_sel(ts, meth, K, num, thres, data_sel_dict ,weights_dict, sel_questions, outdir).select_model()
                     #   intermediate = model_obj.select_model()
                        acc.append(model_obj['test_acc'])
                        
                        list_output_dict.append(model_obj)
    '''Once all the models are run, select the model with best test accuracy and return the output dict for that model.'''
    best_index = np.argmax(acc)
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


if __name__ == "__main__":
    main()
