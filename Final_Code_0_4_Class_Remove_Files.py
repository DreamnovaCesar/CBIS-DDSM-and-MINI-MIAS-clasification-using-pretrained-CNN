
from Final_Code_0_0_Libraries import *
from Final_Code_0_1_Class_Utilities import Utilities

# ? Random remove all files in folder

class RemoveFiles(Utilities):
    """
    Utilities inheritance

    A class used to remove files inside a folder.

    Methods:
        data_dic(): description

        remove_all_files(): description

        remove_random_files(): description

    """

    # ? Decorator to remove all files inside a dir
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

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Keyword Args:
            folder (str): description 
            NFR (int): description
        """

        # * Instance attributes (Protected)
        self._Folder_path = kwargs.get('folder', None);
        self._Number_Files_to_remove = kwargs.get('NFR', None);


    # * Class variables
    def __repr__(self):
            return f'[{self._Folder_path}, {self._Number_Files_to_remove}]';

    # * Class description
    def __str__(self):
        return  f'A class used to remove files inside a folder.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, change format class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self._Folder_path),
                'Number of files to remove': str(self._Number_Files_to_remove),
                };
    
    # * _Folder_path attribute
    @property
    def _Folder_path_property(self):
        return self._Folder_path

    @_Folder_path_property.setter
    def _Folder_path_property(self, New_value):
        self._Folder_path = New_value;
    
    @_Folder_path_property.deleter
    def _Folder_path_property(self):
        print("Deleting Folder_path...");
        del self._Folder_path

    # * _Files_to_remove attribute
    @property
    def _Files_to_remove_property(self):
        return self._Files_to_remove

    @_Files_to_remove_property.setter
    def _Files_to_remove_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Files_to_remove must be a integer") #! Alert
        self._Files_to_remove = New_value;
    
    @_Files_to_remove_property.deleter
    def _Files_to_remove_property(self):
        print("Deleting Files_to_remove...");
        del self._Files_to_remove

    # ? Method to remove all the files inside the dir
    @Utilities.timer_func
    @staticmethod
    def remove_all_files(self) -> None:
        """
        Remove all the files inside the dir

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

    # ? Method to files randomly inside a dir
    @Utilities.timer_func
    def remove_random_files(self) -> None:
        """
        Remove files randomly inside the folder path.

        """

        if(self._Number_Files_to_remove == None and ):
            warnings.warn("Warning...........Message")

        # * This function will remove all the files inside a folder
        Files = os.listdir(self._Folder_path);

            #Filename, Format = os.path.splitext(File)

        for File_sample in sample(Files, self._Number_Files_to_remove):
            print(File_sample);
            #print('Removing: {}{} ✅'.format(Filename, Format));
            os.remove(os.path.join(self._Folder_path, File_sample));