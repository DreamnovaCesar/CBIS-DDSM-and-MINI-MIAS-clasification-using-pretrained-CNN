
from Final_Code_0_0_Libraries import wraps

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import cv2

from Final_Code_0_0_Template_General_Functions import sort_images

from Final_Code_0_1_Class_Utilities import Utilities

class ChangeFormat(Utilities):
    """
    Utilities inheritance

    A class used to change the format to another.

    Methods:
        data_dic(): description

        CropMIAS(): description

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs):
        """
        Keyword Args:
            folder (str): description 
            Newfolder (str): description
            severity (str): description
            Newformat (str):description
        """

        # * General parameters
        self.__Folder = kwargs.get('folder', None)
        self.__New_folder = kwargs.get('newfolder', None)
        self.__Format = kwargs.get('format', None)
        self.__New_format = kwargs.get('newformat', None)

        # * Values, type errors.
        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder attribute must be a string") #! Alert

        if self.__New_folder == None:
            raise ValueError("Folder destination does not exist") #! Alert
        if not isinstance(self.__New_folder, str):
            raise TypeError("Folder destination attribute must be a string") #! Alert

        if self.__Format == None:
            raise ValueError("Current format does not exist") #! Alert
        if not isinstance(self.__Format, str):
            raise TypeError("Current format must be a string") #! Alert

        if self.__New_format == None:
            raise ValueError("New format does not exist") #! Alert
        if not isinstance(self.__New_format, str):
            raise TypeError("Current format must be a string") #! Alert

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}, {self.__New_folder}, {self.__Format}, {self.__New_format}]';

    # * Class description
    def __str__(self):
        return  f'A class used to change the format to another.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, change format class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder),
                'New folder path': str(self.__New_folder),
                'Format': str(self.__Format),
                'New format': str(self.__New_format),
                };

    # * Folder attribute
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

    # * New folder attribute
    @property
    def __New_folder_property(self):
        return self.__New_folder

    @__New_folder_property.setter
    def __New_folder_property(self, New_value):
        self.__New_folder = New_value
    
    @__New_folder_property.deleter
    def __New_folder_property(self):
        print("Deleting folder...")
        del self.__New_folder

    # * Format attribute
    @property
    def __Format_property(self):
        return self.__Format

    @__Format_property.setter
    def __Format_property(self, New_value):
        self.__Format = New_value
    
    @__Format_property.deleter
    def __New_folder_property(self):
        print("Deleting folder...")
        del self.__Format

    # * New Format attribute
    @property
    def __New_format_property(self):
        return self.__New_format

    @__New_format_property.setter
    def __New_format_property(self, New_value):
        self.__New_format = New_value
    
    @__New_format_property.deleter
    def __New_format_property(self):
        print("Deleting new format...")
        del self.__New_format

    # ? Method to change the format, for instance it could be png to jpg
    @Utilities.timer_func
    def ChangeFormat(self):
        """
        Method to change the format.
        """
        # * Changes the current working directory to the given path
        os.chdir(self.__Folder)
        print(os.getcwd())
        print("\n")

        # * Using the sort function
        Sorted_files, Total_images = sort_images(self.__Folder)
        Count:int = 0

        # * Reading the files
        for File in Sorted_files:
            if File.endswith(self.__Format):

                try:
                    Filename, Format  = os.path.splitext(File)
                    print('Working with {} of {} {} images, {} ------- {} ✅'.format(Count, Total_images, self.__Format, Filename, self.__New_format))
                    #print(f"Working with {Count} of {Total_images} {self.Format} images, {Filename} ------- {self.New_format} ✅")
                    
                    # * Reading each image using cv2
                    Path_file = os.path.join(self.__Folder, File)
                    Image = cv2.imread(Path_file)         
                    #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
                    
                    # * Changing its format to a new one
                    New_name_filename = Filename + self.__New_format
                    New_folder = os.path.join(self.__New_folder, New_name_filename)

                    cv2.imwrite(New_folder, Image)
                    #FilenamesREFNUM.append(Filename)
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
                    #print('Cannot convert %s ❌' % File) #! Alert

        print("\n")
        #print(f"COMPLETE {Count} of {Total_images} TRANSFORMED ✅")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert