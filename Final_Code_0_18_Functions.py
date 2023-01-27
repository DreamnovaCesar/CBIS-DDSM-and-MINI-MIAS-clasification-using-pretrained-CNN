# ? Sort Files

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import pd
from Final_Code_0_0_Libraries import warnings
from Final_Code_0_1_Class_Utilities import Utilities

class FunctionsData(Utilities):
    """
    Utilities inheritance

    ######

    Methods:
        data_dic(): description

        sort_images(): description

    """

    # ?
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

    # ?
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

    @staticmethod
    def concat_dataframe(dfs: list[pd.DataFrame], **kwargs: str) -> pd.DataFrame:
        """
        Concat multiple dataframes and name it using technique and the class problem

        Args:
            Dataframe (pd.DataFrames): Multiple dataframes can be entered for concatenation

        Raises:
            ValueError: If the folder variable does not found give a error
            TypeError: _description_
            Warning: _description_
            TypeError: _description_
            Warning: _description_
            TypeError: _description_

        Returns:
            pd.DataFrame: Return the concatenated dataframe
        """
        # * this function concatenate the number of dataframes added

        # * General parameters

        Folder_path = kwargs.get('folder', None)
        Technique = kwargs.get('technique', None)
        Class_problem = kwargs.get('classp', None)
        Save_file = kwargs.get('savefile', False)

        # * Values, type errors and warnings
        if Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(Folder_path, str):
            raise TypeError("Folder attribute must be a string") #! Alert
        
        if Technique == None:
            #raise ValueError("Technique does not exist")  #! Alert
            warnings.warn("Technique does not found, the string 'Without_Technique' will be implemented") #! Alert
            Technique = 'Without_Technique'
        if not isinstance(Technique, str):
            raise TypeError("Technique attribute must be a string") #! Alert

        if Class_problem == None:
            #raise ValueError("Class problem does not exist")  #! Alert
            warnings.warn("Class problem does not found, the string 'No_Class' will be implemented") #! Alert
        if not isinstance(Class_problem, str):
            raise TypeError("Class problem must be a string") #! Alert

        # * Concatenate each dataframe
        #ALL_dataframes = [df for df in dfs]
        #print(len(ALL_dataframes))
        Final_dataframe = pd.concat(dfs, ignore_index = True, sort = False)
            
        #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
        #print(DataFrame)

        # * Name the final dataframe and save it into the given path

        if Save_file == True:
            #Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
            Dataframe_name = '{}_Dataframe_{}.csv'.format(str(Class_problem), str(Technique))
            Dataframe_folder_save = os.path.join(Folder_path, Dataframe_name)
            Final_dataframe.to_csv(Dataframe_folder_save)

        return Final_dataframe