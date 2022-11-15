
from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import pd
from Final_Code_0_0_Libraries import Fernet
from Final_Code_0_0_Libraries import random
from Final_Code_0_0_Libraries import shutil
from Final_Code_0_0_Libraries import datetime

from Final_Code_0_1_Utilities import Utilities

# ? Generate keys

class SecurityFiles(Utilities):

    def __init__(self, **kwargs) -> None:
        
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Number_keys = kwargs.get('NK', 2);
        self.__Key_path = kwargs.get('KP', None);
        self.__Keys_path = kwargs.get('KsP', None);
        self.__Key_chosen = kwargs.get('KC', None);
        self.__Key_random = kwargs.get('KR', None);

    @Utilities.timer_func
    def generate_key(self) -> None: 
        
        # *
        Names = [];
        Keys = [];
        
        # * key generation
        for i in range(self.__Number_keys):

            Key = Fernet.generate_key()
            
            print('Key created: {}'.format(Key))

            Key_name = 'filekey_{}'.format(i)
            Key_path_name = '{}/filekey_{}.key'.format(self.__Folder_path, i)

            Keys.append(Key)
            Names.append(Key_name)

            with open(Key_path_name, 'wb') as Filekey:
                Filekey.write(Key)

            Dataframe_keys = pd.DataFrame({'Name':Names, 'Keys':Keys})

            Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
            Dataframe_Key_folder = os.path.join(self.__Folder_path, Dataframe_Key_name)

            Dataframe_keys.to_csv(Dataframe_Key_folder)

    # ? Encrypt files

    @Utilities.timer_func
    def encrypt_files(self) -> None:

        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        Filenames = []

        if self.__Key_random == True:

            File = random.choice(os.listdir(self.__Keys_path))
            FilenameKey, Format = os.path.splitext(File)

            if File.endswith('.key'):

                try:
                    with open(self.__Keys_path + '/' + File, 'rb') as filekey:
                        Key = filekey.read()

                    Fernet_ = Fernet(Key)

                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(self.__Keys_path, File), self.__Key_chosen)

                    # * This function sort the files and show them

                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename)

                        with open(self.__Folder_path + '/' + Filename, 'rb') as File_: # open in readonly mode
                            Original_file = File_.read()
                        
                        Encrypted_File = Fernet_.encrypt(Original_file)

                        with open(self.__Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File) 

                    with open(self.__Key_path_chosen + '/' + FilenameKey + '.txt', "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(FilenameKey, Filenames))   

                except OSError:
                    print('Is not a key {} ❌'.format(str(File))) #! Alert

        elif self.__Key_random == False:

            Name_key = os.path.basename(self.__Key_path)
            Key_dir = os.path.dirname(self.__Key_path)

            if self.__Key_path.endswith('.key'):
                
                try: 
                    with open(self.__Key_path, 'rb') as filekey:
                        Key = filekey.read()

                    Fernet_ = Fernet(Key)

                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(Key_dir, Name_key), self.__Key_chosen)

                    # * This function sort the files and show them

                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename)

                        with open(self.__Folder_path + '/' + Filename, 'rb') as File: # open in readonly mode
                            Original_file = File.read()
                        
                        Encrypted_File = Fernet_.encrypt(Original_file)

                        with open(self.__Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File)

                    with open(self.__Key_path_chosen + '/' + Name_key + '.txt', "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(Name_key, Filenames))  

                except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert

    # ? Decrypt files

    @Utilities.timer_func
    def decrypt_files(self) -> None: 


        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        Key_dir = os.path.dirname(self.__Key_path)
        Key_file = os.path.basename(self.__Key_path)

        Filename_key, Format = os.path.splitext(Key_file)

        Datetime = datetime.datetime.now()

        with open(self.__Key_path, 'rb') as Filekey:
            Key = Filekey.read()

        Fernet_ = Fernet(Key)

        # * This function sort the files and show them

        if Filename_key.endswith('.key'):

            try:
                for Filename in os.listdir(self.__Folder_path):

                    print(Filename)

                    with open(self.__Folder_path + '/' + Filename, 'rb') as Encrypted_file: # open in readonly mode
                        Encrypted = Encrypted_file.read()
                    
                    Decrypted = Fernet_.decrypt(Encrypted)

                    with open(self.__Folder_path + '/' + Filename, 'wb') as Decrypted_file:
                        Decrypted_file.write(Decrypted)

                with open(Key_dir + '/' + Key_file + '.txt', "w") as text_file:
                        text_file.write('Key used. Datetime: {} '.format(Datetime))  

            except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert