import feature_selection
from sklearn.model_selection import train_test_split

def main():

    
    #using chi2 method to select features
    
    chi2_ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test_chi2")
    chi2_dataset = chi2_ftsel.get_ftsel_original_data()
    X = chi2_dataset[:, :-1]
    y = chi2_dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    chi2_data_dict = {
        'data': chi2_dataset,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    chi2_data_ftsel_dict, chi2_ftsel_questions = chi2_ftsel.ftsel_chi2(chi2_data_dict,KBest = 20)
    print("chi2_ftsel_questions",chi2_ftsel_questions)
    chi2_df = chi2_ftsel.ft_corr(chi2_data_ftsel_dict["X_train"], chi2_ftsel_questions)
    print(chi2_df)
    
    #using mutual information method to select features
    
    mi_ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test_mutlinfo")
    mi_dataset = mi_ftsel.get_ftsel_original_data()
    X = mi_dataset[:, :-1]
    y = mi_dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    mi_data_dict = {
        'data': mi_dataset,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    mi_data_ftsel_dict, mi_ftsel_questions = mi_ftsel.ftsel_mutlinfo(mi_data_dict,KBest = 20)
    print("mi_ftsel_questions",mi_ftsel_questions)
    mi_df = mi_ftsel.ft_corr(mi_data_ftsel_dict["X_train"], mi_ftsel_questions)
    print(mi_df)
    
    #using PCA

    pca_ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test_pca")
    pca_dataset = pca_ftsel.get_ftsel_original_data()
    X = pca_dataset[:, :-1]
    y = pca_dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    pca_data_dict = {
        'data': pca_dataset,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    pca_data_dict, pca_comp_dict = pca_ftsel.ftsel_pca(pca_data_dict,KBest=20)
    pca_df = pca_ftsel.ft_corr(pca_data_dict["X_train"], questions=None)
    print(pca_df) 
    
if __name__ == "__main__":
    main()
