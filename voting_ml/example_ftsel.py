import feature_selection

def main():

    ftsel = feature_selection.FeatureSelection(necess_que_file="../extern/manage_data/list_all_questions.txt", unnecess_que_file="../extern/manage_data/list_unnecessary_columns.txt", bool_necess_que=False, run_name="test")

    dataset = ftsel.get_ftsel_original_data()
    
    data_ftsel, ftsel_questions = ftsel.ftsel_chi2(KBest = 20)

    print("data_ftsel",data_ftsel)
    print("ftsel_questions",ftsel_questions)

    df = ftsel.ft_corr(data_ftsel, ftsel_questions)

    
    
    print(df)
    
if __name__ == "__main__":
    main()
