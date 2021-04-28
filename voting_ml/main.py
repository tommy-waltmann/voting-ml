import numpy as np
import sklearn
import subprocess
from sklearn import model_selection, tree

import data
import feature_selection
import model_sel
import os

import matplotlib.pyplot as plt
import seaborn as sns

def main():

    #parameter space
    list_test_size = [0.2] # decide this
    list_ftsel_method = ['chi2','mutlinfo','pca','dt']
    list_num_features = [20] # decide this
    list_Kfold = [5]
    list_corr_threshold = [1,0.6] # decide this

    param_space = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 10],
        'max_leaf_nodes': [2, 4, 6, 8, 10, 12, 15],
    }

    repeat = 1

    #output dictrionary list
    list_output_dict = []

    # output directory path
    outdir = "../results/run_each_method/"

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
                    data_ranked_dict, ranked_questions = {}, []
                    ftsel_obj =None
                    if(meth=='chi2'):
                        ftsel_obj = feature_selection.FeatureSelection(
                            necess_que_file="../extern/manage_data/list_all_questions.txt",
                            unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                            bool_necess_que=False,
                            outdir=outdir,
                            run_name="chi2"
                        )
                        data_ranked_dict, ranked_questions = ftsel_obj.ftsel_chi2(data_dict, thres)
                    elif(meth=='mutlinfo'):
                        ftsel_obj = feature_selection.FeatureSelection(
                            necess_que_file="../extern/manage_data/list_all_questions.txt",
                            unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                            bool_necess_que=False,
                            outdir=outdir,
                            run_name="mutlinfo"
                        )
                        data_ranked_dict, ranked_questions = ftsel_obj.ftsel_mutlinfo(data_dict, thres)
                    elif(meth=='pca'):
                        ftsel_obj = feature_selection.FeatureSelection(
                            necess_que_file="../extern/manage_data/list_all_questions.txt",
                            unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                            bool_necess_que=False,
                            outdir=outdir,
                            run_name="pca"
                        )
                        data_ranked_dict,_ = ftsel_obj.ftsel_pca(data_dict)
                        fts = data_sel_dict['X_train'].shape[1]
                        questions_int = list(map(str, list(range(1,fts+1,1))))
                        ranked_questions = ["ft_"+x for x in questions_int]
                    elif(meth=='dt'):
                        ftsel_obj = feature_selection.FeatureSelection(
                            necess_que_file="../extern/manage_data/list_all_questions.txt",
                            unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt",
                            bool_necess_que=False,
                            outdir=outdir,
                            run_name="dt"
                        )
                        data_ranked_dict, ranked_questions = ftsel_obj.ftsel_decision_tree_method(data_dict, thres)
                    for num in list_num_features:
                        data_sel_dict, sel_questions = ftsel_obj.select_num_features(data_ranked_dict, num, ranked_questions)
                        ftsel_obj.plot_heatmap(data_sel_dict['X_train'], sel_questions)
                        for K in list_Kfold:
                            '''Here create a class onject of "model_sel" and output all the best parameters and values into "list_output_dict". Then, can create a .csv file to list all the models and accuracies.'''
                            model_obj = model_sel.model_sel(ts, run_num, meth, param_space, K, num, thres, data_sel_dict ,weights_dict, sel_questions, outdir).select_model()
                            #   intermediate = model_obj.select_model()
                            acc.append(model_obj['test_acc'])

                            o_models_file.write(str(ts)+",")
                            o_models_file.write(str(run_num)+",")
                            o_models_file.write(meth+",")
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


if __name__ == "__main__":
    main()
