
from Final_Code_0_0_Libraries import os
from Final_Code_0_0_Libraries import pd
from Final_Code_0_0_Libraries import Fernet
from Final_Code_0_0_Libraries import random
from Final_Code_0_0_Libraries import shutil
from Final_Code_0_0_Libraries import datetime

from Final_Code_0_1_Utilities import Utilities

# ? Generate keys

class SecurityFiles(Utilities):
    """
    Utilities inheritance: A class used to create keys and save the files with it

    Keyword Args:
        folder (str): description 
        NK (int): description
        KP (str): description
        KsP (str): description
        KC (str): description
        KR (bool): description

    Methods:
        generate_key(): description

        encrypt_files(): description

        decrypt_files(): description
    """
    # *Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Number_keys = kwargs.get('NK', 2);
        self.__Key_path = kwargs.get('KP', None);
        self.__Keys_path = kwargs.get('KsP', None);
        self.__Key_chosen = kwargs.get('KC', None);
        self.__Key_random = kwargs.get('KR', None);

    def __repr__(self):
        return f'[{self.__Folder_path}, {self.__Number_keys}, {self.__Key_path}, {self.__Keys_path}, {self.__Key_chosen}, {self.__Key_random}]';

    def __str__(self):
        return  f'Utilities inheritance: A class used to create keys and save the files with it';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Security files class destroyed.');

    def data_dic(self):

        return {'Folder path': str(self.__Folder_path),
                'Number of keys': str(self.__Number_keys),
                'Key path': str(self.__Key_path),
                'Keys path': str(self.__Keys_path),
                'Key chosen': str(self.__Key_chosen),
                'Key random': str(self.__Key_random),
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

    # * __Number_keys attribute
    @property
    def __Number_keys_property(self):
        return self.__Number_keys;

    @__Number_keys_property.setter
    def __Number_keys_property(self, New_value):
        self.__Number_keys = New_value;
    
    @__Number_keys_property.deleter
    def __Number_keys_property(self):
        print("Deleting number of keys...");
        del self.__Number_keys;

    # * __Key_path attribute
    @property
    def __Key_path_property(self):
        return self.__Key_path;

    @__Key_path_property.setter
    def __Key_path_property(self, New_value):
        self.__Key_path = New_value;
    
    @__Key_path_property.deleter
    def __Key_path_property(self):
        print("Deleting key path...");
        del self.__Key_path;

    # * __Keys_path attribute
    @property
    def __Keys_path_property(self):
        return self.__Keys_path;

    @__Keys_path_property.setter
    def __Keys_path_property(self, New_value):
        self.__Keys_path = New_value;
    
    @__Keys_path_property.deleter
    def __Keys_path_property(self):
        print("Deleting keys path...");
        del self.__Keys_path;

    # * __Key_chosen attribute
    @property
    def __Key_chosen_property(self):
        return self.__Key_chosen;

    @__Key_chosen_property.setter
    def __Key_chosen_property(self, New_value):
        self.__Key_chosen = New_value;
    
    @__Key_chosen_property.deleter
    def __Key_chosen_property(self):
        print("Deleting keys path...");
        del self.__Key_chosen;

    # * __Key_random attribute
    @property
    def __Key_random_property(self):
        return self.__Key_random;

    @__Key_random_property.setter
    def __Key_random_property(self, New_value):
        self.__Key_random = New_value;
    
    @__Key_random_property.deleter
    def __Key_random_property(self):
        print("Deleting keys random value...");
        del self.__Key_random;

    @Utilities.timer_func
    def generate_key(self) -> None: 
        """
        Method used to create random keys using self.__Number_keys for interation variable

        """

        # * Create new lists
        Names = [];
        Keys = [];
        
        # * key generation
        for i in range(self.__Number_keys):
            
            # * Generate a new key
            Key = Fernet.generate_key();

            print('Key created: {}'.format(Key));

            # * Name the new key
            Key_name = 'filekey_{}'.format(i)
            Key_path_name = '{}/filekey_{}.key'.format(self.__Folder_path, i);

            # * Append values inside lists
            Keys.append(Key);
            Names.append(Key_name);

            with open(Key_path_name, 'wb') as Filekey:
                Filekey.write(Key);

            # * Create a new dataframe with the follow column names
            Dataframe_keys = pd.DataFrame({'Name':Names, 'Keys':Keys})
            Dataframe_Key_name = 'Dataframe_filekeys.csv'.format();
            Dataframe_Key_folder = os.path.join(self.__Folder_path, Dataframe_Key_name);

            # * Save dataframe in the new path
            Dataframe_keys.to_csv(Dataframe_Key_folder);

    # ? Encrypt files

    @Utilities.timer_func
    def encrypt_files(self) -> None:
        """
        Method used to encrypt files by choosing the key or get it randomly

        """

        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        # * Create new list
        Filenames = [];

        # * Choose option with the bool value (self.__Key_random)
        if self.__Key_random == True:
            
            # * Choose randomly the key inside the (self.__Keys_path)
            File = random.choice(os.listdir(self.__Keys_path));
            FilenameKey, Format = os.path.splitext(File);

            if File.endswith('.key'):

                try:
                    with open('{}/{}'.format(self.__Keys_path, File), 'rb') as filekey:
                        Key = filekey.read();

                    # * Generate a object using Fernet class
                    Fernet_ = Fernet(Key);

                    # * Copy the key
                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(self.__Keys_path, File), self.__Key_chosen);

                    # * This function sort the files and show them
                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename);

                        # * Read the files
                        with open('{}/{}'.format(self.__Folder_path, Filename), 'rb') as File_: # open in readonly mode
                            Original_file = File_.read();
                        
                        # * Encrypt the file sing the object created
                        Encrypted_File = Fernet_.encrypt(Original_file);
                        
                        # * Save the files encrypted already
                        with open('{}/{}'.format(self.__Folder_path, Filename), 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File);
                    
                    # * Save a txt document explaining which files were encrypted and which key could decrypt them
                    with open('{}/{}.txt'.format(self.__Key_path, FilenameKey), "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(FilenameKey, Filenames));

                except OSError:
                    print('Is not a key {} ❌'.format(str(File))) #! Alert

        # * If you want to chose the key
        elif self.__Key_random == False:
            
            # * Split key dir and key name
            Name_key = os.path.basename(self.__Key_path);
            Key_dir = os.path.dirname(self.__Key_path);

            # * if this file is a key
            if self.__Key_path.endswith('.key'):
                
                try: 
                    with open(self.__Key_path, 'rb') as filekey:
                        Key = filekey.read();
                    
                    # * Generate a object using Fernet class
                    Fernet_ = Fernet(Key);

                    # * Copy the key
                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(Key_dir, Name_key), self.__Key_chosen);

                    # * This function sort the files and show them
                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename);

                        # * Read the files
                        with open('{}/{}'.format(self.__Folder_path, Filename), 'rb') as File: # open in readonly mode
                            Original_file = File.read();
                        
                        # * Encrypt the file sing the object created
                        Encrypted_File = Fernet_.encrypt(Original_file)

                        # * Save the files encrypted already
                        with open('{}/{}'.format(self.__Folder_path, Filename), 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File);

                    # * Save a txt document explaining which files were encrypted and which key could decrypt them
                    with open('{}/{}.txt'.format(self.__Key_path, FilenameKey), "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(Name_key, Filenames))  ;

                except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert

    # ? Decrypt files

    @Utilities.timer_func
    def decrypt_files(self) -> None: 
        """
        Method used to decrypt files by using the fenet key

        """
        
        # * Split key dir and key name
        Key_dir = os.path.dirname(self.__Key_path);
        Key_file = os.path.basename(self.__Key_path);

        Filename_key, Format = os.path.splitext(Key_file);

        # * Get the datetime when it was decrypted
        Datetime = datetime.datetime.now();

        with open(self.__Key_path, 'rb') as Filekey:
            Key = Filekey.read();

        # * Generate a object using Fernet class
        Fernet_ = Fernet(Key);

        # * This function sort the files and show them

        if Filename_key.endswith('.key'):

            try:
                for Filename in os.listdir(self.__Folder_path):

                    print(Filename);

                    # * Read the files
                    with open(self.__Folder_path + '/' + Filename, 'rb') as Encrypted_file: # open in readonly mode
                        Encrypted = Encrypted_file.read();
                    
                    # * Decrypt the file sing the object created
                    Decrypted = Fernet_.decrypt(Encrypted);

                    # * Save the files decrypted already
                    with open(self.__Folder_path + '/' + Filename, 'wb') as Decrypted_file:
                        Decrypted_file.write(Decrypted);

                # * Update the txt document showing the datatime when it was decrypted and with which key.
                with open(Key_dir + '/' + Key_file + '.txt', "w") as text_file:
                        text_file.write('Key used. Datetime: {} '.format(Datetime));

            except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert