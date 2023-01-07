# ? Split folders into train/test/validation
from Final_Code_0_0_Libraries import splitfolders
from Final_Code_0_1_Class_Utilities import Utilities

class SplitDataFolder(Utilities):
    """
    Utilities inheritance

    A class used to split data into three stages: training, validation and test.

    Methods:
        data_dic(): description

        split_folders_train_test_val(): Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set, it'll be 80, 10, and 10.

        split_folders_train_test(): Create a new folder with the folders of the class problem and its distribution of training and test.
        The split is 80 and 20.

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs):
        """
        Keyword Args:
            folder (str): description 
        """
        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
    
    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}]';

    # * Class description
    def __str__(self):
        return  f'A class used to split data into three stages: training, validation and test.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, split folders class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder),
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

    # ? Method to split images into training, validations, and test folders.
    @Utilities.timer_func
    def split_folders_train_test_val(self) -> str:
        """
        Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set, it'll be 80, 10, and 10.
        """
        # * General parameters

        Asterisks: int = 50
        Train_split: float = 0.8
        Test_split: float = 0.1
        Validation_split: float = 0.1

        #Name_dir = os.path.dirname(Folder)
        #Name_base = os.path.basename(Folder)
        #New_Folder_name = Folder_path + '_Split'

        # *
        New_folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 

        return New_folder_name

    # ? Method to split images into training, and test folders.
    @Utilities.timer_func
    def split_folders_train_test(self) -> str:
        """
        Create a new folder with the folders of the class problem and its distribution of training and test.
        The split is 80 and 20.

        """
        # * General parameters

        Asterisks: int = 50
        Train_split: float = 0.8
        Test_split: float = 0.2

        #Name_dir = os.path.dirname(Folder)
        #Name_base = os.path.basename(Folder)
        #New_Folder_name = Folder_path + '_Split'

        # *
        New_folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_folder_name, seed = 22, ratio = (Train_split, Test_split)) 

        return New_folder_name