import os
import cv2
import json

from Final_Code_0_0_Libraries import wraps

from Final_Code_0_18_Functions import FunctionsData
from Final_Code_0_1_Class_Utilities import Utilities

class ChangeFormat(Utilities):
    """
    A class used to change the format of a file to another. This class is a subclass of the `Utilities` class.

    Attributes
    ----------
    Folder : str
        The path to the folder containing the files to be changed.
    NewFolder : str
        The path to the folder where the changed files will be saved.
    Format : str
        The current format of the files.
    NewFormat : str
        The desired format for the files.

    Methods
    -------
    data_dic()
        Returns a dictionary containing information about the class properties.
    ChangeFormat()
        A method to change the format of the files.
    """
    # * Initializing (Constructor)
    def __init__(self, **kwargs):
        """
        Keyword Arguments
        -----------------
        Folder : str
            The path to the folder containing the files to be changed.
        NewFolder : str
            The path to the folder where the changed files will be saved.
        Format : str
            The current format of the files.
        NewFormat : str
            The desired format for the files.
        """

        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
        self.__New_folder = kwargs.get('NewFolder', None)
        #self.__Format = kwargs.get('Format', None)
        self.__New_format = kwargs.get('NewFormat', None)


    # * Class variables
    def __repr__(self):
        """
        Return a string representation of the object.

        Returns:
            str: String representation of the object.
        """
        return f'''[{self.__Folder}, 
                    {self.__New_folder}, 
                    {self.__New_format}]''';

    # * Class description
    def __str__(self):
        """
        Return a string description of the object.

        Returns:
            str: String description of the object.
        """
        return  f'A class used to change the format to another.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor called when the object is deleted.
        """
        print('Destructor called, change format class destroyed.');

    # * Get data from a dic
    def data_dic(self) -> dict:
        """
        Return a dictionary containing information about the class properties.

        Returns:
            dict: Dictionary containing information about the class properties.
        """

        return {'Folder': str(self.__Folder),
                'NewFolder': str(self.__New_folder),
                'NewFormat': str(self.__New_format),
                };

    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.

        Returns:
        None
        """
        Data = {'Folder': str(self.__Folder),
                'NewFolder': str(self.__New_folder),
                'NewFormat': str(self.__New_format),
                };

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)

    # * Folder attribute
    @property
    def __Folder_property(self):
        """Getter method for the `Folder` property."""
        return self.__Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        """Setter method for the `Folder` property.

        Args:
            New_value (str): The new value to be assigned to the `folder` attribute.
        """
        self.__Folder = New_value
    
    @__Folder_property.deleter
    def __Folder_property(self):
        """Deleter method for the `Folder` property."""
        print("Deleting folder...")
        del self.__Folder

    # * NewFolder attribute
    @property
    def __New_folder_property(self):
        """Getter method for the `NewFolder` property."""
        return self.__New_folder

    @__New_folder_property.setter
    def __New_folder_property(self, New_value):
        """Setter method for the `Folder` property.

        Args:
            New_value (str): The new value to be assigned to the `NewFolder` attribute.
        """
        self.__New_folder = New_value
    
    @__New_folder_property.deleter
    def __New_folder_property(self):
        """Deleter method for the `NewFolder` property."""
        print("Deleting folder...")
        del self.__New_folder

    '''
    # * Format attribute
    @property
    def __Format_property(self):
        """Getter method for the `Format` property."""
        return self.__Format

    @__Format_property.setter
    def __Format_property(self, New_value):
        """Setter method for the `Format` property.

        Args:
            New_value (str): The new value to be assigned to the `Format` attribute.
        """
        self.__Format = New_value
    
    @__Format_property.deleter
    def __New_folder_property(self):
        """Deleter method for the `NewFolder` property."""
        print("Deleting folder...")
        del self.__Format
    '''
    
    # * NewFormat attribute
    @property
    def __New_format_property(self):
        """Getter method for the `NewFormat` property."""
        return self.__New_format

    @__New_format_property.setter
    def __New_format_property(self, New_value):
        """Setter method for the `NewFormat` property.

        Args:
            New_value (str): The new value to be assigned to the `NewFormat` attribute.
        """
        self.__New_format = New_value
    
    @__New_format_property.deleter
    def __New_format_property(self):
        """Deleter method for the `NewFolder` property."""
        print("Deleting new format...")
        del self.__New_format

    # ? Method to change the format, for instance it could be png to jpg
    @Utilities.timer_func
    def ChangeFormat(self):
        """
        A method to change the format of the images.

        This method changes the format of images from the current format to a new format, specified in the newformat attribute. 
        The method reads the images from the folder specified in the folder attribute, and saves the converted images in the 
        folder specified in the newfolder attribute. The method also tracks and reports the progress of the conversion.
        """
        # * Changes the current working directory to the given path
        os.chdir(self.__Folder)
        print(os.getcwd())
        print("\n")

        # * Using the sort function
        Sorted_files, Total_images = FunctionsData.sort_images(self.__Folder)
        Count:int = 1

        # * Reading the files
        for File in Sorted_files:
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            if File.endswith(Format):

                try:
                    Filename, Format  = os.path.splitext(File)
                    print('Working with {} of {} {} images, {} ------- {} ✅'.format(Count, Total_images, Format, Filename, self.__New_format))
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
        print('{} of {} tranformed ✅. From {} to {}.'.format(Count, Total_images, Format, self.__New_format)) #! Alert