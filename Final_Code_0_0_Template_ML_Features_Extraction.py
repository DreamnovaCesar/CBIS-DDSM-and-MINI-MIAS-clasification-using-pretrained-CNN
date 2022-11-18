from Final_Code_0_0_Libraries import *

from Final_Code_0_0_Template_General_Functions import concat_dataframe
from Final_Code_0_13_Class_Extract_features import FeatureExtraction

def Testing_ML_Models_Biclass_FOF(Model, Technique, All_images, All_labels):

    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1 Score", 
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Biclass' + '_Dataframe_' + 'FOF_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Labels_biclass = ['Mass', 'Calcification']
    #Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_Tumor = All_images[1]

    Labels_Normal = All_labels[0]
    Labels_Tumor = All_labels[1]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_tumor = FeatureExtraction(Images = Images_Tumor, Label = Labels_Tumor)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_first_order_from_images()
    Dataframe_tumor, X_tumor, Y_tumor, Technique_name_tumor = ML_extraction_biclass_tumor.textures_Feature_first_order_from_images()

    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_tumor, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_biclass, Biclass_Data_CSV, Biclass_Data_Models)

    return Dataframe_results

def Testing_ML_Models_Biclass_GLCM(Model, Technique, All_images, All_labels):

    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1 Score", 
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Biclass' + '_Dataframe_' + 'GLCM_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Labels_biclass = ['Mass', 'Calcification']
    #Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_Tumor = All_images[1]

    Labels_Normal = All_labels[0]
    Labels_Tumor = All_labels[1]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_tumor = FeatureExtraction(Images = Images_Tumor, Label = Labels_Tumor)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_GLCM_from_images()
    Dataframe_tumor, X_tumor, Y_tumor, Technique_name_tumor = ML_extraction_biclass_tumor.textures_Feature_GLCM_from_images()
    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_tumor, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_biclass, Biclass_Data_CSV, Biclass_Data_Models)

    return Dataframe_results

def Testing_ML_Models_Biclass_GLRLM(Model, Technique, All_images, All_labels):

    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1 Score", 
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Biclass' + '_Dataframe_' + 'GLRLM_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Labels_biclass = ['Mass', 'Calcification']
    #Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_Tumor = All_images[1]

    Labels_Normal = All_labels[0]
    Labels_Tumor = All_labels[1]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_tumor = FeatureExtraction(Images = Images_Tumor, Label = Labels_Tumor)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_GLRLM_from_images()
    Dataframe_tumor, X_tumor, Y_tumor, Technique_name_tumor = ML_extraction_biclass_tumor.textures_Feature_GLRLM_from_images()
    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_tumor, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_biclass, Biclass_Data_CSV, Biclass_Data_Models)

    return Dataframe_results

def Testing_ML_Models_Multiclass_FOF(Model, Technique, All_images, ALL_labels):


    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1_Score", 
                    "Precision normal", "Recall normal", "F1_Score normal", "Images support normal",
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "Auc Normal", "Auc Benign", "Auc Malignant"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Multiclass' + '_Dataframe_' + 'FOF_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Multiclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    #Labels_biclass = ['Mass', 'Calcification']
    Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_benign = All_images[1]
    Images_malignant = All_images[2]

    Labels_Normal = ALL_labels[0]
    Labels_benign = ALL_labels[1]
    Labels_malignant = ALL_labels[2]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_benign = FeatureExtraction(Images = Images_benign, Label = Labels_benign)
    ML_extraction_biclass_malignant = FeatureExtraction(Images = Images_malignant, Label = Labels_malignant)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_first_order_from_images()
    Dataframe_benign, X_benign, Y_benign, Technique_name_benign = ML_extraction_biclass_benign.textures_Feature_first_order_from_images()
    Dataframe_malignant, X_malignant, Y_malignant, Technique_name_malignant = ML_extraction_biclass_malignant.textures_Feature_first_order_from_images()

    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_benign, Dataframe_malignant, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_triclass, Multiclass_Data_CSV, Multiclass_Data_Models)

    return Dataframe_results
    
def Testing_ML_Models_Multiclass_GLCM(Model, Technique, All_images, ALL_labels):


    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1_Score", 
                    "Precision normal", "Recall normal", "F1_Score normal", "Images support normal",
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "Auc Normal", "Auc Benign", "Auc Malignant"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Multiclass' + '_Dataframe_' + 'GLRLM_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Multiclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    #Labels_biclass = ['Mass', 'Calcification']
    Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_benign = All_images[1]
    Images_malignant = All_images[2]

    Labels_Normal = ALL_labels[0]
    Labels_benign = ALL_labels[1]
    Labels_malignant = ALL_labels[2]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_benign = FeatureExtraction(Images = Images_benign, Label = Labels_benign)
    ML_extraction_biclass_malignant = FeatureExtraction(Images = Images_malignant, Label = Labels_malignant)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_GLCM_from_images()
    Dataframe_benign, X_benign, Y_benign, Technique_name_benign = ML_extraction_biclass_benign.textures_Feature_GLCM_from_images()
    Dataframe_malignant, X_malignant, Y_malignant, Technique_name_malignant = ML_extraction_biclass_malignant.textures_Feature_GLCM_from_images()

    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_benign, Dataframe_malignant, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_triclass, Multiclass_Data_CSV, Multiclass_Data_Models)

    return Dataframe_results

def Testing_ML_Models_Multiclass_GLRLM(Model, Technique, All_images, ALL_labels):


    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1_Score", 
                    "Precision normal", "Recall normal", "F1_Score normal", "Images support normal",
                    "Precision Mass", "Recall Mass", "F1_Score Mass", "Images support Mass",
                    "Precision Calcification", "Recall Calcification", "F1_Score Calcification", "Images support Calcification",
                    "Precision macro avg", "Recall macro avg", "F1_Score macro avg", "Images support macro avg",
                    "Precision weighted avg", "Recall weighted avg", "F1_Score weighted avg", "Images support weighted avg",
                    "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "Auc Normal", "Auc Benign", "Auc Malignant"]

    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Multiclass' + '_Dataframe_' + 'GLRLM_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Multiclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    #Labels_biclass = ['Mass', 'Calcification']
    Labels_triclass = ['Normal', 'Mass', 'Calcification']

    Images_Normal = All_images[0]
    Images_benign = All_images[1]
    Images_malignant = All_images[2]

    Labels_Normal = ALL_labels[0]
    Labels_benign = ALL_labels[1]
    Labels_malignant = ALL_labels[2]
        
    ML_extraction_biclass_normal = FeatureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_benign = FeatureExtraction(Images = Images_benign, Label = Labels_benign)
    ML_extraction_biclass_malignant = FeatureExtraction(Images = Images_malignant, Label = Labels_malignant)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_GLRLM_from_images()
    Dataframe_benign, X_benign, Y_benign, Technique_name_benign = ML_extraction_biclass_benign.textures_Feature_GLRLM_from_images()
    Dataframe_malignant, X_malignant, Y_malignant, Technique_name_malignant = ML_extraction_biclass_malignant.textures_Feature_GLRLM_from_images()

    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_benign, Dataframe_malignant, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = Technique, SaveCSV = False)

    Dataframe_results = Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Technique_name_normal, Labels_triclass, Multiclass_Data_CSV, Multiclass_Data_Models)

    return Dataframe_results