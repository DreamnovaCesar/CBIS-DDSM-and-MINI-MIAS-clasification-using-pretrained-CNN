# ? Sort Files

from Final_Code_0_0_Libraries import os
from Final_Code_0_1_Class_Utilities import Utilities

class SortData(Utilities):
    """
    Utilities inheritance

    ######

    Methods:
        data_dic(): description

        sort_images(): description

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs):
        """
        Keyword Args:
            folder (str): description 
        """

        # * General parameters
        self.__Folder = kwargs.get('folder', None)

        # * Values, type errors.
        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder attribute must be a string") #! Alert

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}]';

    # * Class description
    def __str__(self):
        return  f'A class used to change the format to another.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, change format class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder)};

    @staticmethod
    def sort_images(Folder_path: str) -> tuple[list[str], int]: 
        """
        Sort the filenames of the obtained folder path.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            list[str]: Return all files sorted.
            int: Return the number of images inside the folder.
        """
        
        # * Folder attribute (ValueError, TypeError)
        if Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        Asterisks:int = 60
        # * This function sort the files and show them

        Number_images: int = len(os.listdir(Folder_path))
        print("\n")
        print("*" * Asterisks)
        print('Images: {}'.format(Number_images))
        print("*" * Asterisks)
        Files: list[str] = os.listdir(Folder_path)
        print("\n")

        Sorted_files: list[str] = sorted(Files)

        for Index, Sort_file in enumerate(Sorted_files):
            print('Index: {} ---------- {} ✅'.format(Index, Sort_file))

        print("\n")

        return Sorted_files, Number_images