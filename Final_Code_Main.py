from Final_Code_0_Libraries import *

from Final_Code_1_General_Functions import BarChart
from Final_Code_1_General_Functions import split_folders_train_test_val

from Final_Code_5_CNN_Architectures import *

from Final_Code_CBIS_DDSM_4_Data_Augmentation import preprocessing_DataAugmentation_Folder

#Model_CNN = (Model_pretrained, Model_pretrained)


def plot_data():
    

    Data_show = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Accuracy training FE', label = "Percentage", column = 3, reverse = False, classes = 2)

    Data_show1 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Accuracy training LE', label = "Percentage", column = 4, reverse = False, classes = 2)
    
    Data_show2 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Accuracy testing', label = "Percentage", column = 5, reverse = False, classes = 2)
    
    Data_show3 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Overall AUC', label = "Percentage", column = 38, reverse = False, classes = 2)
    
    Data_show4 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'F1-score', label = "Percentage", column = 13, reverse = False, classes = 2)
    
    Data_show5 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Precision', label = "Percentage", column = 11, reverse = False, classes = 2)

    Data_show6 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Recall', label = "Percentage", column = 12, reverse = False, classes = 2)

    Data_show7 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Training loss', label = "Percentage", column = 6, reverse = True, classes = 2)

    Data_show8 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Testing loss', label = "Percentage", column = 7, reverse = True, classes = 2)

    Data_show9 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Training time', label = "Seconds", column = 30, reverse = True, classes = 2)

    Data_show10 = BarChart(csv = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_CNN_Folder_CLAHE_NO.csv", 
                            foldersave = r'D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataModels', title = 'Testing time', label = "Seconds", column = 31, reverse = True, classes = 2)

    Data_show.barchart_horizontal()
    Data_show1.barchart_horizontal()
    Data_show2.barchart_horizontal()
    Data_show3.barchart_horizontal()
    Data_show4.barchart_horizontal()
    Data_show5.barchart_horizontal()
    Data_show6.barchart_horizontal()
    Data_show7.barchart_horizontal()
    Data_show8.barchart_horizontal()
    Data_show9.barchart_horizontal()
    Data_show10.barchart_horizontal()

def plot_data_ML():

    #CSV = r"D:\Mini-MIAS\Mini-MIAS Final\Multiclass_Data_CSV\Multiclass_Dataframe_All_Techniques_FOF.csv"
    CSV = r"D:\Mini-MIAS\Mini-MIAS Final\Multiclass_Data_CSV\Multiclass_Dataframe_All_Techniques_GLCM.csv"
    ML_folder = r"D:\Mini-MIAS\Mini-MIAS Final\Multiclass_Data_Models"

    Name = 'GLCM'

    Data_show = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'Accuracy', label = "Percentage", column = 3, reverse = False, classes = 3, name = Name )

    Data_show1 = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'Precision', label = "Percentage", column = 4, reverse = False, classes = 3, name = Name )

    Data_show2 = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'Recall', label = "Percentage", column = 5, reverse = False, classes = 3, name = Name )

    Data_show3 = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'F1-score', label = "Percentage", column = 6, reverse = False, classes = 3, name = Name )

    Data_show4 = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'Time training', label = "Seconds", column = 9, reverse = False, classes = 3, name = Name )

    Data_show5 = BarChart(csv = CSV, 
                            foldersave = ML_folder, 
                                title = 'AUC', label = "Percentage", column = 18, reverse = False, classes = 3, name = Name )
    
    Data_show.barchart_horizontal()
    Data_show1.barchart_horizontal()
    Data_show2.barchart_horizontal()
    Data_show3.barchart_horizontal()
    Data_show4.barchart_horizontal()
    Data_show5.barchart_horizontal()

def main():
    #plot_data_ML()
    #Testing_CNN_Models_Biclass_From_Folder(Model_CNN, 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass' + '_Split', 'TEST')

    Model_CNN = (13, 14)

    #Bic = 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass'
    #Multic = "D:\Mini-MIAS\CBIS_DDSM_NO_Images_Multiclass"
    
    #MulticSplit = split_folders_train_test_val(Multic, False)

    #preprocessing_DataAugmentation_Folder(MulticSplit, ['A', 'B', 'C'], [1, 1, 22])

    configuration_models_folder(folder = "D:\Mini-MIAS\CBIS_DDSM_NO_Images_Multiclass" + '_Split', foldermodels = 'D:\Test',
                                    foldermodelesp = 'D:\Test', foldercsv = 'D:\Test', 
                                        models = Model_CNN, technique = 'TEST', 
                                            labels = ['A', 'B', 'C'], X = 224, Y = 224, epochs = 2)
    

if __name__ == "__main__":
    main()