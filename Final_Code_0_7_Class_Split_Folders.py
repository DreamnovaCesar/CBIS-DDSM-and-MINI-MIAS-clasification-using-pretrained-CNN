# ? Split folders into train/test/validation
from Final_Code_0_0_Libraries import splitfolders
from Final_Code_0_1_Class_Utilities import Utilities

class SplittingData(Utilities):
    """
    A class used to split data into three stages: training, validation and test.

    Inherits:
        Utilities: a utility class that provides some helper methods.

    Methods:
        data_dic(): Get data from a dictionary.

        split_folders_train_test_val(): Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set, it'll be 80, 10, and 10.
        If not, it'll be 80, and 20 (Training, and test).

    Attributes:
        __Folder (str): The folder path.

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Initializes SplittingData with the specified arguments.

        Args:
            **kwargs (dict): Keyword arguments:
                - Folder (str): The folder path.

        Returns:
            None.

        """
        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
    
    # * Class variables
    def __repr__(self):
        """
        Return a string representation of the object.

        Returns:
            str: String representation of the object.
        """
        return f'[{self.__Folder}]';

    # * Class description
    def __str__(self):
        """
        Return a string description of the object.

        Returns:
            str: String description of the object.
        """
        return  f'A class used to split data into three stages: training, validation and test.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor called when the object is deleted.
        """
        print('Destructor called, split folders class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        """
        Return a dictionary containing information about the class properties.

        Returns:
            dict: Dictionary containing information about the class properties.

        """

        return {'Folder': str(self.__Folder),
                };

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

    # ? Method to split images into training, validations, and test folders.
    @Utilities.timer_func
    def split_folders_train_test_val(self, Val: bool = True) -> None:
        """
        Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set is activate, it'll be 80, 10, and 10 (Training, test and validation).
        If not, it'll be 80, and 20 (Training, and test).
        """

        try:
            
            # * General parameters
            Asterisks: int = 50
            Train_split: float = 0.8

            # * Concatenate '_Split' with the folder given.
            New_folder_name = '{}_Split'.format(self.__Folder)

            print("*" * Asterisks)
            print('New folder name: {}'.format(New_folder_name))
            print("*" * Asterisks)

            if(Val):
                
                Test_split: float = 0.1
                Validation_split: float = 0.1

                # * Split the folder (Training, validation, and test.)
                splitfolders.ratio(self.__Folder, output = New_folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 

            elif(not Val):
                 
                Test_split: float = 0.2

                # * Split the folder (Training, validation, and test.)
                splitfolders.ratio(self.__Folder, output = New_folder_name, seed = 22, ratio = (Train_split, Test_split)) 

            else:
                pass;

        except OSError:
                print('The value of Val given is {}. Val must be True or False ❌'.format(Val)); #! Alert
