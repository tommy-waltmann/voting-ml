import feature_selection
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    
    dt_ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test_dec_tree")
    dataframe = pd.DataFrame(dt_ftsel.get_ftsel_original_data(), columns= dt_ftsel._ftsel_quelist)
    
    dt_data_ftsel_dict, dt_ftsel_questions = dt_ftsel.ftsel_decision_tree_method(dataframe, input_test_size = 0.3, num_features = 20)
    print("dt_ftsel_questions",dt_ftsel_questions)
    
    
if __name__ == "__main__":
    main()