
from Final_Code_0_0_Libraries import wraps

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import sample

from Final_Code_0_1_Utilities import Utilities

# ? Random remove all files in folder

class RemoveFiles(Utilities):

    # ? Remove all files inside a dir
    @staticmethod
    @Utilities.timer_func 
    def remove_all_files(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * 
            for File in os.listdir(self._Folder_path):
                Filename, Format  = os.path.splitext(File);
                print('Removing: {} . {} ✅'.format(Filename, Format));
                os.remove(os.path.join(self._Folder_path, File));

            result = func(self, *args, **kwargs)

            return result
        return wrapper

    def __init__(self, **kwargs) -> None:

        # * Instance attributes (Protected)
        self._Folder_path = kwargs.get('folder', None);
        self._Number_Files_to_remove = kwargs.get('NFR', None);

    def __repr__(self):

        kwargs_info = "[{}, {}]".format(self._Folder_path, self._Number_Files_to_remove);

        return kwargs_info

    def __str__(self):
        pass
    
    # * Folder_path attribute
    @property
    def Folder_path_property(self):
        return self._Folder_path

    @Folder_path_property.setter
    def Folder_path_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder_path must be a string") #! Alert
        self._Folder_path = New_value;
    
    @Folder_path_property.deleter
    def Folder_path_property(self):
        print("Deleting Folder_path...");
        del self._Folder_path

    # * Files_to_remove attribute
    @property
    def Files_to_remove_property(self):
        return self._Files_to_remove

    @Files_to_remove_property.setter
    def Files_to_remove_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Files_to_remove must be a integer") #! Alert
        self._Files_to_remove = New_value;
    
    @Files_to_remove_property.deleter
    def Files_to_remove_property(self):
        print("Deleting Files_to_remove...");
        del self._Files_to_remove

    # ? Remove all files inside a dir
    @Utilities.timer_func
    def remove_all_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """
        
        # * Folder attribute (ValueError, TypeError)
        if self._Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self._Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        # * This function will remove all the files inside a folder
        for File in os.listdir(self._Folder_path):
            Filename, Format  = os.path.splitext(File);
            print('Removing: {} . {} ✅'.format(Filename, Format));
            os.remove(os.path.join(self._Folder_path, File));

    # ? Remove all files inside a dir
    @Utilities.timer_func
    def remove_random_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """

        # * This function will remove all the files inside a folder
        Files = os.listdir(self._Folder_path);

            #Filename, Format = os.path.splitext(File)

        for File_sample in sample(Files, self._Number_Files_to_remove):
            print(File_sample);
            #print('Removing: {}{} ✅'.format(Filename, Format));
            os.remove(os.path.join(self._Folder_path, File_sample));