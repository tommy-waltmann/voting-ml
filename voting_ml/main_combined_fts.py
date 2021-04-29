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
    list_test_size = [0.2]#,0.15,0.2] # decide this
    list_ftsel_method = ['chi2','mutlinfo','pca','dt']
    list_run_names = ['test_chi2', 'test_mutlinfo', 'test_pca', 'test_dt']
    list_Kfold = [5]#,5]
    list_corr_threshold = [1, 0.6] # decide this
    repeat = 5
    param_space = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 3, 4, 5],#, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
            'max_leaf_nodes': [2, 4, 6, 8, 10]#, 12, 15],
    }
    #output dictrionary list
    list_output_dict = []
    
    # options for different runs
    1. Use unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt"
    outdir = "../results/run_combined_1_union_fts/"
    '''2. Use unnecess_que_file="../extern/manage_data/list_2_unnecessary_columns.txt"
    outdir = "../results/run_combined_2_union_fts/"
    3. Use unnecess_que_file="../extern/manage_data/list_3_unnecessary_columns.txt"
    outdir = "../results/run_combined_3_union_fts/"
    4. Use unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt"
    outdir = "../results/run_combined_4_union_fts/"
    5. Use unnecess_que_file="../extern/manage_data/list_5_unnecessary_columns.txt"
    outdir = "../results/run_combined_5_union_fts/"
    '''

    if(not os.path.isdir(outdir)):
        os.mkdir(outdir) 

    o_models_file = open(outdir+"models.csv","w")
    o_models_file.write("test size,run num,ftsel method,Kfold,number of features,correlation threshold,best features,criterion,max_depth,max_leaf_nodes,min_samples_leaf,min_samples_split,training accuracy,test accuracy\n")
        
    #splitting data and weights into train, test (refer to optimal_params.py)
    poll_data = data.PollDataProxy(remove_nan=True, convert_to_float=True)
    
    acc = []
    
    '''refer to optimal_params.py. Functions from this python scripts are transferred here. (get_bad_questions() and separate_weights().)'''
    
    for ts in list_test_size:
        for run_num in range(repeat):    
            all_data, all_data_questions = poll_data.all_data_except(get_bad_questions())
            X = all_data[:, :-1]
            y = all_data[:, -1]
            X_train_neg, X_test_neg, y_train, y_test = model_selection.train_test_split(X, y,
                                                                                test_size=ts,
                                                                                shuffle=True)
            X_train_neg, weights_train, questions = separate_weights(X_train_neg, all_data_questions[:-1])
            X_test_neg, weights_test, _ = separate_weights(X_test_neg, all_data_questions[:-1])

            X_train = np.where(X_train_neg == -1, 0, X_train_neg)
            X_test = np.where(X_test_neg == -1, 0, X_test_neg)

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
                chi2_ftsel = feature_selection.FeatureSelection(
                    necess_que_file="../extern/manage_data/list_all_questions.txt",
                    unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt",
                    bool_necess_que=False,
                    outdir=outdir+"fts_"+str(ts)+"_"+str(run_num)+"_"+str(thres)+"/",
                    run_name="chi2"
                )
                chi2_data_dict, chi2_questions = chi2_ftsel.ftsel_chi2(data_dict, thres)
                
                mutlinfo_ftsel = feature_selection.FeatureSelection(
                    necess_que_file="../extern/manage_data/list_all_questions.txt",
                    unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt",
                    bool_necess_que=False,
                    outdir=outdir+"fts_"+str(ts)+"_"+str(run_num)+"_"+str(thres)+"/",
                    run_name="mutlinfo"
                )
                mutlinfo_data_dict, mutlinfo_questions = mutlinfo_ftsel.ftsel_mutlinfo(data_dict, thres)
                
                pca_ftsel = feature_selection.FeatureSelection(
                    necess_que_file="../extern/manage_data/list_all_questions.txt",
                    unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt",
                    bool_necess_que=False,
                    outdir=outdir+"fts_"+str(ts)+"_"+str(run_num)+"_"+str(thres)+"/",
                    run_name="pca"
                )
                pca_data_dict,_ = pca_ftsel.ftsel_pca(data_dict)
                
                fts = pca_data_dict['X_train'].shape[1]
                questions_int = list(map(str, list(range(1,fts+1,1))))
                pca_questions = ["ft_"+x for x in questions_int]
                
                dt_ftsel = feature_selection.FeatureSelection(
                    necess_que_file="../extern/manage_data/list_all_questions.txt",
                    unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt",
                    bool_necess_que=False,
                    outdir=outdir+"fts_"+str(ts)+"_"+str(run_num)+"_"+str(thres)+"/",
                    run_name="dt"
                )
                dt_data_dict, dt_questions = dt_ftsel.ftsel_decision_tree_method(data_dict, thres)
                
                #print("chi2_questions",chi2_questions)
                #print("mutlinfo_questions",mutlinfo_questions)
                #print("dt_questions",dt_questions)
                
                chi2_data_sel_dict, chi2_sel_questions = chi2_ftsel.select_num_features(chi2_data_dict, 7, chi2_questions)
                mutlinfo_data_sel_dict, mutlinfo_sel_questions = mutlinfo_ftsel.select_num_features(mutlinfo_data_dict, 7, mutlinfo_questions)
                dt_data_sel_dict, dt_sel_questions = dt_ftsel.select_num_features(dt_data_dict, 7, dt_questions)
                
                union_selected_questions = union_elements(chi2_sel_questions, mutlinfo_sel_questions, dt_sel_questions)
                
                union_X_train = df_X_train.filter(items=union_selected_questions).to_numpy()
                union_X_test = df_X_test.filter(items=union_selected_questions).to_numpy()
                
                union_sel_data_dict = {
                    'X_train' : union_X_train,
                    'X_test' : union_X_test,
                    'y_train' : y_train,
                    'y_test' : y_test
                }
                
                print("union questions:",union_selected_questions)
                print("union_Sel_data_dict",union_sel_data_dict)
                
                chi2_ftsel.plot_heatmap(union_sel_data_dict['X_train'], union_selected_questions)
                
                for K in list_Kfold:
                    #Here create a class onject of "model_sel" and output all the best parameters and values into "list_output_dict". Then, can create a .csv file to list all the models and accuracies.
                    model_obj = model_sel.model_sel(ts, run_num, 'combined', param_space, K, len(union_selected_questions), thres, union_sel_data_dict ,weights_dict, union_selected_questions, outdir).select_model()
                    
                    acc.append(model_obj['test_acc'])
                    
                    o_models_file.write(str(ts)+",")
                    o_models_file.write(str(run_num)+",")
                    o_models_file.write("combined,")
                    o_models_file.write(str(K)+",")
                    o_models_file.write(str(len(union_selected_questions))+",")
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
    f = open("../extern/manage_data/list_4_unnecessary_columns.txt", 'r')
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

def union_elements(list1,list2,list3):
    s1 = set(list1)
    s2 = set(list2)
    s3 = set(list3)

    set1 = s1.union(s2)
    result_set = set1.union(s3)

    final_list = list(result_set)
    return final_list
'''
def same_features(thres, data_dict, list_ftsel_method, listlist_run_names):
    nested_sel_questions = []
    for meth,test in zip(list_ftsel_method, listlist_run_names):
        if test != 'test_pca':
            ftsel_obj = feature_selection.FeatureSelection(
                       necess_que_file="../extern/manage_data/list_all_questions.txt",
                       unnecess_que_file="../extern/manage_data/list_4_unnecessary_columns.txt",
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
