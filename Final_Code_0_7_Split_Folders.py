# ? Split folders into train/test/validation

from Final_Code_0_0_Libraries import splitfolders
from Final_Code_0_1_Utilities import Utilities

class SplitDataFolder(Utilities):

    # * Change the format of one image to another 

    def __init__(self, **kwargs):

        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
    
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

    @Utilities.timer_func
    def split_folders_train_test_val(self) -> str:
        """
        Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set, it'll be 80, 10, and 10.

        Args:
            Folder_path (str): Folder's dataset for distribution

        Returns:
            None
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
        New_Folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_Folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 

        return New_Folder_name

    def split_folders_train_test(self) -> str:

        """
        Create a new folder with the folders of the class problem and its distribution of training and test.
        The split is 80 and 20.

        Args:
            Folder_path (str): Folder's dataset for distribution

        Returns:
            None
        """
        # * General parameters

        Asterisks: int = 50
        Train_split: float = 0.8
        Test_split: float = 0.2

        #Name_dir = os.path.dirname(Folder)
        #Name_base = os.path.basename(Folder)
        #New_Folder_name = Folder_path + '_Split'

        # *
        New_Folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_Folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split)) 

        return New_Folder_name