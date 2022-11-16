# ?

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import pd
from Final_Code_0_0_Libraries import random
from Final_Code_0_0_Libraries import randint

from Final_Code_0_1_Utilities import Utilities

class Generator(Utilities):
    """
    Utilities inheritance

    A class used to generate folders and data from students

    Methods:
        create_folders(): description

        creating_data_students(): description

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Keyword Args:
            folder (str): description 
            FN (list): description
            iter (int): description
            SD (bool): description
        """
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Folders_name = kwargs.get('FN', None);
        self.__Iteration = kwargs.get('iter', None);
        self.__Save_dataframe = kwargs.get('SD', None);

    # * Class variables
    def __repr__(self):
        return f'[{self.__Folder_path}, {self.__Folders_name}, {self.__Iteration}, {self.__Save_dataframe}]';

    # * Class description
    def __str__(self):
        return  f'A class used to generate folders and data from students';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Generator class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder_path),
                'Folders names': str(self.__Folders_name),
                'Iteration': str(self.__Iteration),
                'Dataframe': str(self.__Save_dataframe),
                };

    # * __Folder_path attribute
    @property
    def __Folder_path_property(self):
        return self.__Folder_path;

    @__Folder_path_property.setter
    def __Folder_path_property(self, New_value):
        self.__Folder_path = New_value;
    
    @__Folder_path_property.deleter
    def __Folder_path_property(self):
        print("Deleting folder path...");
        del self.__Folder_path;
    
    # * __Folders_name attribute
    @property
    def __Folders_name_property(self):
        return self.__Folders_name;

    @__Folders_name_property.setter
    def __Folders_name_property(self, New_value):
        self.__Folders_name = New_value;
    
    @__Folders_name_property.deleter
    def __Folders_name_property(self):
        print("Deleting folder name or folder names..");
        del self.__Folders_name;

    # * __Iteration attribute
    @property
    def __Iteration_property(self):
        return self.__Iteration;

    @__Iteration_property.setter
    def __Iteration_property(self, New_value):
        self.__Iteration = New_value;
    
    @__Iteration_property.deleter
    def __Iteration_property(self):
        print("Deleting iteration values..");
        del self.__Iteration;

    # * __Save_dataframe
    @property
    def __Save_dataframe_property(self):
        return self.__Save_dataframe;

    @__Save_dataframe_property.setter
    def __Save_dataframe_property(self, New_value):
        self.__Save_dataframe = New_value;
    
    @__Save_dataframe_property.deleter
    def __Save_dataframe_property(self):
        print("Deleting save dataframe value..");
        del self.__Save_dataframe;

    # ? Method to create folders
    @Utilities.timer_func
    def create_folders(self) -> pd.DataFrame: 
        """
        Method used to create folders 

        """

        # * Create the list to save names and absolute dir
        Path_names = [];
        Path_absolute_dir = [];

        # * if multiple folders name in the list are included
        if(len(self.__Folders_name) >= 2):
            
            # * Enumerate each one
            for i, Path_name in enumerate(self.__Folders_name):

                # * Combine the name with the dir
                Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name[i]);
                print(Folder_path_new);

                # * Append names and absolute dir
                Path_names.append(Path_name);
                Path_absolute_dir.append(Folder_path_new);

                # * Check if the dir is already created
                Exist_dir = os.path.isdir(Folder_path_new) ;

            # * if not create the new folder
            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name[i]));

        else:

            # * Combine the name with the dir
            Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name);
            print(Folder_path_new);
            
            # * Append names and absolute dir
            Path_names.append(self.__Folders_name);
            Path_absolute_dir.append(Folder_path_new);

            # * Check if the dir is already created
            Exist_dir = os.path.isdir(Folder_path_new) ;

            # * if not create the new folder
            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name));

        # * Save the name inside a new dataframe
        if(self.__Save_dataframe == True):
            Dataframe = pd.DataFrame({'Names':Path_names, 'Path names':Path_absolute_dir});
            Dataframe_name = 'Dataframe_path_names.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);

            #Exist_dataframe = os.path.isfile(Dataframe_folder)

            Dataframe.to_csv(Dataframe_folder);

        return Dataframe

    # ? Method to create folders
    @Utilities.timer_func
    def creating_data_students(self) -> pd.DataFrame: 
        """
        Method used to create random students data 

        """

        # * Tuples for random generation.
        Random_Name = ('Tom', 'Nick', 'Chris', 'Jack', 'Thompson');
        Random_Classroom = ('A', 'B', 'C', 'D', 'E');
        
        # * Column names to create the DataFrame
        Columns_names = ['Name', 'Age', 'Classroom', 'Height', 'Math', 'Chemistry', 'Physics', 'Literature'];

        # * Create a new dataframe using the headers
        Dataframe = pd.DataFrame(Columns_names)

        for i in range(self.__Iteration):

            # * Create a new row randomly
            New_row = { 'Name':random.choice(Random_Name),
                        'Age':randint(16, 26),
                        'Classroom':random.choice(Random_Classroom),
                        'Height':randint(160, 195),
                        'Math':randint(70, 100),
                        'Chemistry':randint(70, 100),
                        'Physics':randint(70, 100),
                        'Literature':randint(70, 100)};

            Dataframe = Dataframe.append(New_row, ignore_index = True);

            # * Show each iteration
            print('Iteration complete: {}'.format(i));

        # * Save the name inside a new dataframe
        if(self.__Save_dataframe == True):
            Dataframe_name = 'Dataframe_students_data.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);
            Dataframe.to_csv(Dataframe_folder);

        return Dataframe