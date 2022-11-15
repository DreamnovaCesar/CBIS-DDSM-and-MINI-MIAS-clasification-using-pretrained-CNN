
from Final_Code_0_0_Libraries import wraps

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import cv2

from Final_Code_1_General_Functions import sort_images

from Final_Code_0_1_Utilities import Utilities

class ChangeFormat(Utilities):
    """
    _summary_

    _extended_summary_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
    """
    # * Change the format of one image to another 

    def __init__(self, **kwargs):
        """
        _summary_

        _extended_summary_

        Raises:
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
        """
        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
        self.__New_folder = kwargs.get('Newfolder', None)
        self.__Format = kwargs.get('Format', None)
        self.__New_format = kwargs.get('Newformat', None)

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

    # * Folder attribute
    @property
    def Folder_property(self):
        return self.__Folder

    @Folder_property.setter
    def Folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.__Folder = New_value
    
    @Folder_property.deleter
    def Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    # * New folder attribute
    @property
    def New_folder_property(self):
        return self.__New_folder

    @New_folder_property.setter
    def New_folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.__New_folder = New_value
    
    @New_folder_property.deleter
    def New_folder_property(self):
        print("Deleting folder...")
        del self.__New_folder

    # * Format attribute
    @property
    def Format_property(self):
        return self.__Format

    @Format_property.setter
    def Format_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Format must be a string") #! Alert
        self.__Format = New_value
    
    @Format_property.deleter
    def New_folder_property(self):
        print("Deleting folder...")
        del self.__Format

    # * New Format attribute
    @property
    def New_format_property(self):
        return self.__New_format

    @New_format_property.setter
    def New_format_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("New format must be a string") #! Alert
        self.__New_format = New_value
    
    @New_format_property.deleter
    def New_format_property(self):
        print("Deleting new format...")
        del self.__New_format

    @Utilities.timer_func
    def ChangeFormat(self):
        """
        _summary_

        _extended_summary_
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