# ? Class for images cropping.

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import cv2
from Final_Code_0_0_Libraries import pd

from Final_Code_1_General_Functions import sort_images

from Final_Code_0_1_Utilities import Utilities

class CropImages(Utilities):
    """
    Utilities inheritance

    A class used to crop Mini-MIAS images using the coordinates from the website.

    Methods:
        data_dic(): description

        CropMIAS(): description

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Keyword Args:
            folder (str): description 
            NF (str): (Normal folder)
            TF (str): (Tumor folder)
            BF (str): (Benign folder)
            MF (str): (Malignant folder)
            Dataframe (pd.dataframe): description
            Shapes (int): description
            X mean (int): description
            Y mean (int): description
        """

        # * This algorithm outputs crop values for images based on the coordinates of the CSV file.
        # * General parameters
        self.__Folder: str = kwargs.get('folder', None)
        self.__Normalfolder: str = kwargs.get('NF', None)
        self.__Tumorfolder: str = kwargs.get('TF', None)
        self.__Benignfolder: str = kwargs.get('BF', None)
        self.__Malignantfolder: str = kwargs.get('MF', None)

        # * CSV to extract data
        self.__Dataframe: pd.DataFrame = kwargs.get('Dataframe', None)
        self.__Shapes = kwargs.get('Shapes', None)
        
        # * X and Y mean to extract normal cropped images
        self.__X_mean:int = kwargs.get('Xmean', None)
        self.__Y_mean:int = kwargs.get('Ymean', None)

        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Normalfolder == None:
            raise ValueError("Folder for normal images does not exist") #! Alert
        if not isinstance(self.__Normalfolder , str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Tumorfolder == None:
            raise ValueError("Folder for tumor images does not exist") #! Alert
        if not isinstance(self.__Tumorfolder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Benignfolder == None:
            raise ValueError("Folder for benign images does not exist") #! Alert
        if not isinstance(self.__Benignfolder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Malignantfolder == None:
            raise ValueError("Folder for malignant images does not exist") #! Alert
        if not isinstance(self.__Malignantfolder, str):
            raise TypeError("Folder must be a string") #! Alert
        
        #elif self.Dataframe == None:
        #raise ValueError("The dataframe is required") #! Alert

        elif self.__Shapes == None:
            raise ValueError("The shape is required") #! Alert

        elif self.__X_mean == None:
            raise ValueError("X_mean is required") #! Alert

        elif self.__Y_mean == None:
            raise ValueError("Y_mean is required") #! Alert

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}, {self.__Normalfolder}, {self.__Tumorfolder}, {self.__Benignfolder}, {self.__Malignantfolder}, {self.__Dataframe}, {self.__Shapes}, {self.__X_mean}, {self.__Y_mean}]';

    # * Class description
    def __str__(self):
        return  f'A class used to crop Mini-MIAS images using the coordinates from the website';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, crop images class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder),
                'Normal folder path': str(self.__Normalfolder),
                'Tumor folder path': str(self.__Tumorfolder),
                'Benign folder path': str(self.__Benignfolder),
                'Malignant folder path': str(self.__Malignantfolder),
                'Dataframe': str(self.__Dataframe),
                'Shapes': str(self.__Shapes),
                'X mean': str(self.__X_mean),
                'Y mean': str(self.__Y_mean),
                };

    # * __Folder attribute
    @property
    def __Folder_property(self):
        return self.__Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        self.__Folder = New_value
    
    @__Folder_property.deleter
    def __Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    # * __Normalfolder attribute
    @property
    def __Normalfolder_property(self):
        return self.__Normalfolder

    @__Normalfolder_property.setter
    def __Normalfolder_property(self, New_value):
        self.__Normalfolder = New_value
    
    @__Normalfolder_property.deleter
    def __Normalfolder_property(self):
        print("Deleting normal folder...")
        del self.__Normalfolder

    # * __Tumorfolder attribute
    @property
    def __Tumorfolder_property(self):
        return self.__Tumorfolder

    @__Tumorfolder_property.setter
    def __Tumorfolder_property(self, New_value):
        self.__Tumorfolder = New_value
    
    @__Tumorfolder_property.deleter
    def __Tumorfolder_property(self):
        print("Deleting tumor folder...")
        del self.__Tumorfolder

    # * __Benignfolder attribute
    @property
    def __Benignfolder_property(self):
        return self.__Benignfolder

    @__Benignfolder_property.setter
    def __Benignfolder_property(self, New_value):
        self.__Benignfolder = New_value
    
    @__Benignfolder_property.deleter
    def __Benignfolder_property(self):
        print("Deleting benign folder...")
        del self.__Benignfolder

    # * __Malignantfolder attribute
    @property
    def __Malignantfolder_property(self):
        return self.__Malignantfolder

    @__Malignantfolder_property.setter
    def __Malignantfolder_property(self, New_value):
        self.__Malignantfolder = New_value
    
    @__Malignantfolder_property.deleter
    def __Malignantfolder_property(self):
        print("Deleting malignant folder...")
        del self.__Malignantfolder

    # * __Dataframe attribute
    @property
    def __Dataframe_property(self):
        return self.__Dataframe

    @__Dataframe_property.setter
    def __Dataframe_property(self, New_value):
        self.__Dataframe = New_value
    
    @__Dataframe_property.deleter
    def __Dataframe_property(self):
        print("Deleting dataframe...")
        del self.__Dataframe

    # * __Shapes attribute
    @property
    def __Shapes_property(self):
        return self.__Shapes

    @__Shapes_property.setter
    def __Shapes_property(self, New_value):
        self.__Shapes = New_value
    
    @__Shapes_property.deleter
    def __Shapes_property(self):
        print("Deleting shapes...")
        del self.__Shapes

    # * __X_mean attribute
    @property
    def __X_mean_property(self):
        return self.__X_mean

    @__X_mean_property.setter
    def __X_mean_property(self, New_value):
        self.__X_mean = New_value
    
    @__X_mean_property.deleter
    def __X_mean_property(self):
        print("Deleting X mean...")
        del self.__X_mean

    # * __Y_mean attribute
    @property
    def __Y_mean_property(self):
        return self.__Y_mean

    @__Y_mean_property.setter
    def __Y_mean_property(self, New_value):
        self.__Y_mean = New_value
    
    @__Y_mean_property.deleter
    def __Y_mean_property(self):
        print("Deleting Y mean...")
        del self.__Y_mean

    # ? Method to crop Mini-MIAS images.
    @Utilities.timer_func
    def CropMIAS(self) -> None:
        
        #Images = []

        os.chdir(self.__Folder)

        # * Columns
        Name_column = 0
        Severity = 3
        X_column = 4
        Y_column = 5
        Radius = 6

        # * Labels
        Benign = 0
        Malignant = 1
        Normal = 2

        # * Initial index
        Index = 1
        
        # * Using sort function
        Sorted_files, Total_images = sort_images(self.__Folder)
        Count = 1

        # * Reading the files
        for File in Sorted_files:
        
            Filename, Format = os.path.splitext(File)

            print("******************************************")
            print(self.__Dataframe.iloc[Index - 1, Name_column])
            print(Filename)
            print("******************************************")

            if self.__Dataframe.iloc[Index - 1, Severity] == Benign:
                if self.__Dataframe.iloc[Index - 1, X_column] > 0  or self.__Dataframe.iloc[Index - 1, Y_column] > 0:
                
                    try:
                    
                        print(f"Working with {Count} of {Total_images} {Format} Benign images, {Filename} X: {self.__Dataframe.iloc[Index - 1, X_column]} Y: {self.__Dataframe.iloc[Index - 1, Y_column]}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        # * Reading the image
                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)
                        
                        #Distance = self.Shape # X and Y.
                        #Distance = self.Shape # Perimetro de X y Y de la imagen.
                        #Image_center = Distance / 2 
                            
                        # * Obtaining the center using the radius
                        Image_center = self.__Dataframe.iloc[Index - 1, Radius] / 2 
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])
                        print(self.__Dataframe.iloc[Index - 1, Radius])

                        # * Extract the value of X and Y of each image
                        X_size = self.__Dataframe.iloc[Index - 1, X_column]
                        print(X_size)
                        Y_size = self.__Dataframe.iloc[Index - 1, Y_column]
                        print(Y_size)
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Benig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        print(Image.shape, " ----------> ", Cropped_Image_Benig.shape)

                        # print(Cropped_Image_Benig.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)

                        New_name_filename = Filename + '_Benign_cropped' + Format

                        New_folder = os.path.join(self.__Benignfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Benig)

                        New_folder = os.path.join(self.__Tumorfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Benig)

                        #Images.append(Cropped_Image_Benig)

                    except OSError:
                            print('Cannot convert %s' % File)

            elif self.__Dataframe.iloc[Index - 1, Severity] == Malignant:
                if self.__Dataframe.iloc[Index - 1, X_column] > 0  or self.__Dataframe.iloc[Index - 1, Y_column] > 0:

                    try:

                        print(f"Working with {Count} of {Total_images} {Format} Malignant images, {Filename} X {self.__Dataframe.iloc[Index - 1, X_column]} Y {self.__Dataframe.iloc[Index - 1, Y_column]}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        # * Reading the image
                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)
                        
                        #Distance = self.Shape # X and Y.
                        #Distance = self.Shape # Perimetro de X y Y de la imagen.
                        #Image_center = Distance / 2 
                            
                        # * Obtaining the center using the radius
                        Image_center = self.__Dataframe.iloc[Index - 1, Radius] / 2 # Center
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])

                        # * Extract the value of X and Y of each image
                        X_size = self.__Dataframe.iloc[Index - 1, X_column]
                        Y_size = self.__Dataframe.iloc[Index - 1, Y_column]
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Malig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        print(Image.shape, " ----------> ", Cropped_Image_Malig.shape)
                
                        # print(Cropped_Image_Malig.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)

                        New_name_filename = Filename + '_Malignant_cropped' + Format

                        New_folder = os.path.join(self.__Malignantfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Malig)

                        New_folder = os.path.join(self.__Tumorfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Malig)

                        #Images.append(Cropped_Image_Malig)


                    except OSError:
                        print('Cannot convert %s' % File)
            
            elif self.__Dataframe.iloc[Index - 1, Severity] == Normal:
                if self.__Dataframe.iloc[Index - 1, X_column] == 0  or self.__Dataframe.iloc[Index - 1, Y_column] == 0:

                    try:

                        print(f"Working with {Count} of {Total_images} {Format} Normal images, {Filename}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)

                        Distance = self.__Shapes # Perimetro de X y Y de la imagen.
                        Image_center = Distance / 2 # Centro de la imagen.
                        #CD = self.df.iloc[Index - 1, Radius] / 2
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])

                        # * Extract the value of X and Y of each image
                        X_size = self.__X_mean
                        Y_size = self.__Y_mean
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Normal = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        # * Comparison two images
                        print(Image.shape, " ----------> ", Cropped_Image_Normal.shape)

                        # print(Cropped_Image_Normal.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)
                    
                        New_name_filename = Filename + '_Normal_cropped' + Format

                        New_folder = os.path.join(self.__Normalfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Normal)

                        #Images.append(Cropped_Image_Normal)

                    except OSError:
                        print('Cannot convert %s' % File)

            Index += 1   