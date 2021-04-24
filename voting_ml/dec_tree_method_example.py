import data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import feature_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns

def main():
    dt_ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test_dec_tree")

    dt_dataset = dt_ftsel.get_ftsel_original_data()
    X = dt_dataset[:, :-1]
    y = dt_dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dt_data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    dt_data_ftsel_dict, dt_ftsel_questions = dt_ftsel.ftsel_decision_tree_method(dt_data_dict,0.5)
    dt_data_sel_dict, dt_sel_questions = dt_ftsel.select_num_features(dt_data_ftsel_dict, 20, dt_ftsel_questions)

    X_train = dt_data_sel_dict['X_train']
    X_test = dt_data_sel_dict['X_test']
    
    # prepare input data
    X_train_enc, X_test_enc, oe = dt_ftsel.prepare_inputs(X_train, X_test)
        
    # prepare output data
    y_train_enc, y_test_enc, le = dt_ftsel.prepare_targets(y_train, y_test)

    model = DecisionTreeClassifier()
    model.fit(X_train_enc, y_train_enc)
    print('The score of the Decision tree classifier using the best features from the decision tree itself is', model.score(X_test_enc,y_test_enc))
    
    df = pd.DataFrame(X_train, columns = dt_ftsel_questions)
    le=LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        df_corr = df.corr(method='pearson')
    plt.figure(figsize=(17,17))
    sns.heatmap(df_corr,linewidths=.1,cmap="YlGnBu", annot=True)
    plt.yticks(rotation=0)
    plt.clf()
    
if __name__ == "__main__":
    main()
