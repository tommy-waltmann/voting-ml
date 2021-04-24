import numpy as np
import sklearn
import subprocess
from sklearn import model_selection, tree

import data
import feature_selection
import model_sel
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    #parameter space
    list_test_size = [0.1]#,0.15,0.2] # decide this
    list_ftsel_method = ['chi2','mutlinfo','pca','dt']
    list_run_names = ['test_chi2', 'test_mutlinfo', 'test_pca', 'test_dt']
    #list_num_features = [10,15,20] # decide this
    list_num_features = [50]
    list_Kfold = [5]#,5]
    list_corr_threshold = [1]#, 0.5, 0.6, 0.7] # decide this
    repeat = 1
    param_space = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 3, 4, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
            'max_leaf_nodes': [2, 4, 6, 8, 10, 12, 15],
    }
    #output dictrionary list
    list_output_dict = []

    # output directory path
    outdir = "../results/run_combined/"
    
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir) 

    o_models_file = open(outdir+"models.csv","w")
    o_models_file.write("test size,run num,ftsel method,Kfold,number of features,correlation threshold,best features,criterion,max_depth,max_leaf_nodes,min_samples_leaf,min_samples_split,training accuracy,test accuracy\n")
        
    #splitting data and weights into train, test (refer to optimal_params.py)
    poll_data = data.PollDataProxy(remove_nan=False, convert_to_float=False)
    
    acc = []
    
    '''refer to optimal_params.py. Functions from this python scripts are transferred here. (get_bad_questions() and separate_weights().)'''
    for run_num in range(repeat):
        for ts in list_test_size:
            
            all_data, all_data_questions = poll_data.all_data_except(get_bad_questions())
            X = all_data[:, :-1]
            y = all_data[:, -1]
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                                test_size=ts,
                                                                                shuffle=True)
            X_train, weights_train, questions = separate_weights(X_train, all_data_questions[:-1])
            X_test, weights_test, _ = separate_weights(X_test, all_data_questions[:-1])

            print("questions",questions)
            
            df_X_train = pd.DataFrame(X_train, columns = questions)
            df_X_test = pd.DataFrame(X_test, columns = questions)

            print("df_X_train",df_X_train)
            print("df_X_test",df_X_test)
            
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

            print("X_train",X_train)
            print("X_test",X_test)
            
            
            #Create class objects of the current selection method
            for thres in list_corr_threshold:
                data_sel_dict, sel_questions = {}, []
                ftsel = feature_selection.FeatureSelection(
                    necess_que_file="../extern/manage_data/list_all_questions.txt",
                    unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                    bool_necess_que=False,
                    run_name="test_ftsel_combined"
                )
                chi2_data_dict, chi2_questions = ftsel.ftsel_chi2(data_dict, thres)
                print(chi2_questions)
                mutlinfo_data_dict, mutlinfo_questions = ftsel.ftsel_mutlinfo(data_dict, thres)
                print(mutlinfo_questions)
                pca_data_dict,_ = ftsel.ftsel_pca(data_dict)
                fts = pca_data_dict['X_train'].shape[1]
                questions_int = list(map(str, list(range(1,fts+1,1))))
                pca_questions = ["ft_"+x for x in questions_int]
                
                dt_data_dict, dt_questions = ftsel.ftsel_decision_tree_method(data_dict, thres)
                print(dt_questions)
                #print("chi2_questions",chi2_questions)
                #print("mutlinfo_questions",mutlinfo_questions)
                #print("dt_questions",dt_questions)
                
                for num in list_num_features:

                    chi2_data_sel_dict, chi2_sel_questions = ftsel.select_num_features(chi2_data_dict, num, chi2_questions)
                    mutlinfo_data_sel_dict, mutlinfo_sel_questions = ftsel.select_num_features(mutlinfo_data_dict, num, mutlinfo_questions)
                    dt_data_sel_dict, dt_sel_questions = ftsel.select_num_features(dt_data_dict, num, dt_questions)
                    
                    same_selected_questions = same_elements(chi2_sel_questions, mutlinfo_sel_questions, dt_sel_questions)
                    common_X_train = df_X_train.filter(items=same_selected_questions).to_numpy()
                    common_X_test = df_X_test.filter(items=same_selected_questions).to_numpy()
                    
                    same_sel_data_dict = {
                        'X_train' : common_X_train.astype(str),
                        'X_test' : common_X_test.astype(str),
                        'y_train' : y_train.astype(str),
                        'y_test' : y_test.astype(str)
                    }

                    print("same questions:",same_selected_questions)
                    print("same_Sel_data_dict",same_sel_data_dict)

                    ftsel.plot_heatmap(same_sel_data_dict['X_train'], same_selected_questions)
                    
                    for K in list_Kfold:
                        #Here create a class onject of "model_sel" and output all the best parameters and values into "list_output_dict". Then, can create a .csv file to list all the models and accuracies.
                        model_obj = model_sel.model_sel(ts, run_num, 'combined', param_space, K, len(same_selected_questions), thres, same_sel_data_dict ,weights_dict, same_selected_questions, outdir).select_model()
                        
                        acc.append(model_obj['test_acc'])

                        o_models_file.write(str(ts)+",")
                        o_models_file.write(str(run_num)+",")
                        o_models_file.write("combined,")
                        o_models_file.write(str(K)+",")
                        o_models_file.write(str(num)+",")
                        o_models_file.write(str(thres)+",")
                        for ii in range(len(model_obj['best_features'])):
                            o_models_file.write(model_obj['best_features'][ii]+" ")
                        o_models_file.write(",")
                        o_models_file.write(model_obj['best_params']['criterion']+",")
                        o_models_file.write(str(model_obj['best_params']['max_depth'])+",")
                        o_models_file.write(str(model_obj['best_params']['max_leaf_nodes'])+",")
                        o_models_file.write(str(model_obj['best_params']['min_samples_leaf'])+",")
                        o_models_file.write(str(model_obj['best_params']['min_samples_split'])+",")
                        o_models_file.write(str(model_obj['train_acc'])+",")
                        o_models_file.write(str(model_obj['test_acc'])+",")
                        o_models_file.write("\n")
                        
                        list_output_dict.append(model_obj)
                        
    '''Once all the models are run, select the model with best test accuracy and return the output dict for that model.'''

    o_models_file.close()
    
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
    new_questions = column_names
    new_questions.remove('weight')
    return new_X_train, weights, new_questions

def same_elements(list1,list2,list3):
    s1 = set(list1)
    s2 = set(list2)
    s3 = set(list3)
      
    # Calculates intersection of 
    # sets on s1 and s2
    set1 = s1.intersection(s2) 
      
    # Calculates intersection of sets
    # on set1 and s3
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    return final_list
'''
def same_features(thres, data_dict, list_ftsel_method, listlist_run_names):
    nested_sel_questions = []
    for meth,test in zip(list_ftsel_method, listlist_run_names):
        if test != 'test_pca':
            ftsel_obj = feature_selection.FeatureSelection(
                       necess_que_file="../extern/manage_data/list_all_questions.txt",
                       unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                       bool_necess_que=False,
                       run_name=test
                       )
            data_sel_dict, sel_questions = ftsel_obj.ftsel_chi2(data_dict, thres)
            nested_sel_questions.append(sel_questions)
                    
    same_sel_questions = same_elements(*nested_sel_questions)
    return same_sel_questions
'''        
if __name__ == "__main__":
    main()
