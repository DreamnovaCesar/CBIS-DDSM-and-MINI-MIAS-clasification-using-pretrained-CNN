# ?

from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import pd
from Final_Code_0_0_Libraries import random
from Final_Code_0_0_Libraries import randint

from Final_Code_0_1_Utilities import Utilities

class Generator(Utilities):

    def __init__(self, **kwargs) -> None:

        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Folders_name = kwargs.get('FN', None);
        self.__Iteration = kwargs.get('iter', None);
        self.__Save_dataframe = kwargs.get('SD', None);

    # ? Create folders
    @Utilities.timer_func
    def create_folders(self) -> pd.DataFrame: 
        """
        _summary_

        _extended_summary_

        Args:
            Folder_path (str): _description_
            Folder_name (list[str]): _description_
            CSV_name (str): _description_
        """

        # *
        Path_names = [];
        Path_absolute_dir = [];

        # *
        if(len(self.__Folders_name) >= 2):
            
            # *
            for i, Path_name in enumerate(self.__Folders_name):

                # *
                Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name[i]);
                print(Folder_path_new);

                Path_names.append(Path_name);
                Path_absolute_dir.append(Folder_path_new);

                Exist_dir = os.path.isdir(Folder_path_new) ;

            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name[i]));

        else:

            # *
            Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name);
            print(Folder_path_new);

            Path_names.append(self.__Folders_name);
            Path_absolute_dir.append(Folder_path_new);

            Exist_dir = os.path.isdir(Folder_path_new) ;

            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name));

        # *
        if(self.__Save_dataframe == True):
            Dataframe = pd.DataFrame({'Names':Path_names, 'Path names':Path_absolute_dir});
            Dataframe_name = 'Dataframe_path_names.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);

            #Exist_dataframe = os.path.isfile(Dataframe_folder)

            Dataframe.to_csv(Dataframe_folder);

        return Dataframe

    # ? Create folders
    @Utilities.timer_func
    def creating_data_students(self) -> pd.DataFrame: 
        
        # * Tuples for random generation.
        Random_Name = ('Tom', 'Nick', 'Chris', 'Jack', 'Thompson');
        Random_Classroom = ('A', 'B', 'C', 'D', 'E');
        
        # * Column names to create the DataFrame
        Columns_names = ['Name', 'Age', 'Classroom', 'Height', 'Math', 'Chemistry', 'Physics', 'Literature'];

        # *
        Dataframe = pd.DataFrame(Columns_names)

        for i in range(self.__Iteration):

            # *
            New_row = { 'Name':random.choice(Random_Name),
                        'Age':randint(16, 26),
                        'Classroom':random.choice(Random_Classroom),
                        'Height':randint(160, 195),
                        'Math':randint(70, 100),
                        'Chemistry':randint(70, 100),
                        'Physics':randint(70, 100),
                        'Literature':randint(70, 100)};

            Dataframe = Dataframe.append(New_row, ignore_index = True);

            # *
            print('Iteration complete: {}'.format(i));

        # *
        if(self.__Save_dataframe == True):
            Dataframe_name = 'Dataframe_students_data.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);
            Dataframe.to_csv(Dataframe_folder);

        return Dataframe