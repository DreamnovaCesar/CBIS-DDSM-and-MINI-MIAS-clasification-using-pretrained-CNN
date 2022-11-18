

from Final_Code_0_0_Template_General_Functions import CBIS_DDSM_CSV_severity_labeled
from Final_Code_0_0_Template_General_Functions import CBIS_DDSM_split_data

def preprocessing_Split_Calc_Mass() -> None:

    Severity = 9

    Calcification = 'Calficication'
    Mass = 'Mass'

    Test = 'Test'
    Training = 'Training'

    Calc_Description_Test = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Calcification\CBIS-DDSM_Calcification_Test\calc_case_description_test_set.csv"
    Calc_Description_Training = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Calcification\CBIS-DDSM_Calcification_Training\calc_case_description_train_set.csv"

    Mass_Description_Test = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Mass\CBIS-DDSM_Mass_Test\mass_case_description_test_set.csv"
    Mass_Description_Training = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Mass\CBIS-DDSM_Mass_Training\mass_case_description_train_set.csv"

    Dataframe_Calc_Severity_Test = CBIS_DDSM_CSV_severity_labeled(Calc_Description_Test, Severity, 1)
    Dataframe_Calc_Severity_Training = CBIS_DDSM_CSV_severity_labeled(Calc_Description_Training, Severity, 1)

    Dataframe_Mass_Severity_Test = CBIS_DDSM_CSV_severity_labeled(Mass_Description_Test, Severity, 2)
    Dataframe_Mass_Severity_Training = CBIS_DDSM_CSV_severity_labeled(Mass_Description_Training, Severity, 2)

    #Folder_CSV, Folder, Folder_total_benign, Folder_benign, Folder_benign_wc, Folder_malignant, Folder_abnormal, Dataframe, Severity, Phase
    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Calc_Severity_Test, Calcification, Test)
    
    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Calc_Severity_Training, Calcification, Training)

    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Mass_Severity_Test, Mass, Test)
    
    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Mass_Severity_Training, Mass, Training)


    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Calc_Severity_Test, Calcification, Test)
    
    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Calc_Severity_Training, Calcification, Training)

    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Mass_Severity_Test, Mass, Test)
    
    Dataframe = CBIS_DDSM_split_data(   '', '', '', '', '', '', '', 
                                        Dataframe_Mass_Severity_Training, Mass, Training)

    ########## ########## ########## ########## ########## ########## ########## ########## ########## ##########