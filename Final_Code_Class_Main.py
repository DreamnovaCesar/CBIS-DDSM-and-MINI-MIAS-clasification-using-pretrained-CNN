from Final_Code_0_0_Libraries import *

from Final_Code_0_7_Class_Split_Folders import SplitDataFolder
from Final_Code_0_9_Class_Figure_Adjust import BarChart
from Final_Code_0_1_Class_Utilities import Utilities

from Final_Code_0_11_Class_CNN_Architectures import *

from Final_Code_0_11_Class_CNN_Architectures import ConfigurationCNN
from Final_Code_0_15_Class_ML_Configuration import ConfigurationML

from Final_Code_CBIS_DDSM_4_Data_Augmentation import preprocessing_DataAugmentation_Folder
from Final_Code_Menu_Tkinter import MenuTkinter

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

def Test():
    
    #plot_data_ML()
    #Testing_CNN_Models_Biclass_From_Folder(Model_CNN, 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass' + '_Split', 'TEST')

    Model_CNN = [1, 2, 3, 4, 5, 6, 7]

    #Bic = r"D:\CBIS-DDSM\CBIS-DDSM Final\CBIS_DDSM_MIAS_CLAHE_Images_Biclass"
    #Multic = r"D:\Mini-MIAS\CBIS_DDSM_NO_Images_Multiclass"

    #BicSplit = SplitDataFolder.split_folders_train_test_val(Bic, False)

    #preprocessing_DataAugmentation_Folder(BicSplit, ['Abnormal', 'Normal'], [2, 12])

    #CNN = ConfigurationCNN(folder = BicSplit, foldermodels = 'D:\Test', foldermodelesp = 'D:\Test', foldercsv = 'D:\Test', 
    #                       models = Model_CNN, technique = 'TEST', labels = ['Abnormal', 'Normal'], X = 224, Y = 224, epochs = 5)
    
    #CNN.configuration_models_folder_CNN()

    ML = ConfigurationML()
    df = ML.Features_extraction_ML(r"D:\Mini-MIAS\Mini-MIAS\CS", 'D:\Test', ['1', '2'], 'GLRLM')

    ML = ConfigurationML(folder = r'D:\Test', dataframe = df, labels = ['1', '2'], EXT = 'GLRLM', technique = 'CS', models = Model_CNN, epochs = 100)
    ML.configuration_models_folder_ML()
    #CLAHE_test = r"D:\CBIS-DDSM\CBIS-DDSM Final\2_Biclass_DataCSV\Biclass_Dataframe_Gray-Level Co-Occurance Matrix_CLAHE.csv"
    
# ?
def main():
    """Main function
    """

    Test()

    #config = MenuTkinter()
    #config.menu()

# ?
if __name__ == "__main__":
    main()