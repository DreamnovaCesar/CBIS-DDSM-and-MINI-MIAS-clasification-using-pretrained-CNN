from Final_Code_0_0_Libraries import *

from functools import wraps

################################################## ? Class decorators

class GeneralUtilities(object):

    @staticmethod  
    def time_func(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * Obtain the executed time of the function
            t1 = time.time()
            result = func(self, *args, **kwargs)
            t2 = time.time()

            print("\n")
            print("*" * 60)
            print('Function {} executed in {:.4f}'.format(func.__name__, t2 - t1))
            print("*" * 60)
            print("\n")

            return result
        return wrapper

# ? Decorator
def asterisk_row_print(func):
     
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def wrapper(*args, **kwargs):
 
        # storing time before function execution
        print("\n")
        print("*" * 30)
         
        func(*args, **kwargs)
 
        # storing time after function execution
        print("*" * 30)
        print("\n")
 
    return wrapper

# ? Decorator

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        #print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        print("\n")
        print("*" * 60)

        print('Function {} executed in {:.4f}'.format(func.__name__, t2 - t1))

        print("*" * 60)
        print("\n")

        return result
    return wrapper

# ? Detect fi GPU exist in your PC for CNN Decorator

def detect_GPU(func) -> None:
    """
    This function shows if a gpu device is available and its name. This function is good if the training is using a GPU  

    Args:
        None

    Returns:
        None
    """

    def wrapper(*args, **kwargs):

      func(*args, **kwargs)

      GPU_name: string = tf.test.gpu_device_name()
      GPU_available: list = tf.config.list_physical_devices()
      print("\n")
      print(GPU_available)
      print("\n")
      #if GPU_available == True:
          #print("GPU device is available")

      if "GPU" not in GPU_name:
          print("GPU device not found")
          print("\n")
      print('Found GPU at: {}'.format(GPU_name))
      print("\n")

    return wrapper

################################################## ? Decorators

# ? Create folders

@timer_func
def create_folders(Folder_path: str, Folder_name: str, CSV_name: str) -> None: 

  Path_names = []
  Path_absolute_dir = []

  if(len(Folder_name) >= 2):

    for i, Path_name in enumerate(Folder_name):

      Folder_path_new = r'{}\{}'.format(Folder_path, Folder_name[i])
      print(Folder_path_new)

      Path_names.append(Path_name)
      Path_absolute_dir.append(Folder_path_new)

      Exist_dir = os.path.isdir(Folder_path_new) 

      if Exist_dir == False:
        os.mkdir(Folder_path_new)
      else:
        print('Path: {} exists, use another name for it'.format(Folder_name[i]))

  else:

    Folder_path_new = r'{}\{}'.format(Folder_path, Folder_name)
    print(Folder_path_new)

    Path_names.append(Folder_name)
    Path_absolute_dir.append(Folder_path_new)

    Exist_dir = os.path.isdir(Folder_path_new) 

    if Exist_dir == False:
      os.mkdir(Folder_path_new)
    else:
      print('Path: {} exists, use another name for it'.format(Folder_name))

  Dataframe_name = 'Dataframe_path_names_{}.csv'.format(CSV_name)
  Dataframe_folder = os.path.join(Folder_path, Dataframe_name)

  #Exist_dataframe = os.path.isfile(Dataframe_folder)

  Dataframe = pd.DataFrame({'Names':Path_names, 'Path names':Path_absolute_dir})
  Dataframe.to_csv(Dataframe_folder)

# ? Create folders

@timer_func
def creating_data_students(Dataframe: pd.DataFrame, Iter: int, Folder_path: str, Save_dataframe: bool = False) -> pd.DataFrame: 
    
    # * Tuples for random generation.
    Random_Name = ('Tom', 'Nick', 'Chris', 'Jack', 'Thompson')
    Random_Classroom = ('A', 'B', 'C', 'D', 'E')

    for i in range(Iter):

        # *
        New_row = {'Name':random.choice(Random_Name),
                   'Age':randint(16, 21),
                   'Classroom':random.choice(Random_Classroom),
                   'Height':randint(160, 190),
                   'Math':randint(70, 100),
                   'Chemistry':randint(70, 100),
                   'Physics':randint(70, 100),
                   'Literature':randint(70, 100)}

        Dataframe = Dataframe.append(New_row, ignore_index = True)

        # *
        print('Iteration complete: {}'.format(i))

    # *
    if(Save_dataframe == True):
      Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
      Dataframe_Key_folder = os.path.join(Folder_path, Dataframe_Key_name)

      Dataframe.to_csv(Dataframe_Key_folder)

    return Dataframe

# ? Generate keys

@timer_func
def generate_key(Folder_path: str, Number_keys: int = 2) -> None: 

    Names = []
    Keys = []
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # key generation
    for i in range(Number_keys):

        Key = Fernet.generate_key()
        
        print('Key created: {}'.format(Key))

        Key_name = 'filekey_{}'.format(i)
        Key_path_name = '{}/filekey_{}.key'.format(Folder_path, i)

        Keys.append(Key)
        Names.append(Key_name)

        with open(Key_path_name, 'wb') as Filekey:
            Filekey.write(Key)

        Dataframe_keys = pd.DataFrame({'Name':Names, 'Keys':Keys})

        Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
        Dataframe_Key_folder = os.path.join(Folder_path, Dataframe_Key_name)

        Dataframe_keys.to_csv(Dataframe_Key_folder)

# ? Encrypt files

@timer_func
def encrypt_files(**kwargs) -> None:

    # * General parameters
    Folder_path = kwargs.get('folderpath', None)
    Key_path = kwargs.get('keypath', None)
    Keys_path = kwargs.get('keyspath', None)
    Key_path_chosen = kwargs.get('newkeypath', None)
    Random_key = kwargs.get('randomkey', False)

    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Filenames = []

    if Random_key == True:

        File = random.choice(os.listdir(Keys_path))
        FilenameKey, Format = os.path.splitext(File)

        if File.endswith('.key'):

            try:
                with open(Keys_path + '/' + File, 'rb') as filekey:
                    Key = filekey.read()

                Fernet_ = Fernet(Key)

                #Key_name = 'MainKey.key'.format()
                shutil.copy2(os.path.join(Keys_path, File), Key_path_chosen)

                # * This function sort the files and show them

                for Filename in os.listdir(Folder_path):

                    Filenames.append(Filename)

                    with open(Folder_path + '/' + Filename, 'rb') as File_: # open in readonly mode
                        Original_file = File_.read()
                    
                    Encrypted_File = Fernet_.encrypt(Original_file)

                    with open(Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                        Encrypted_file.write(Encrypted_File) 

                with open(Key_path_chosen + '/' + FilenameKey + '.txt', "w") as text_file:
                    text_file.write('The key {} open the next documents {}'.format(FilenameKey, Filenames))   

            except OSError:
                print('Is not a key {} ❌'.format(str(File))) #! Alert

    elif Random_key == False:

        Name_key = os.path.basename(Key_path)
        Key_dir = os.path.dirname(Key_path)

        if Key_path.endswith('.key'):
            
            try: 
                with open(Key_path, 'rb') as filekey:
                    Key = filekey.read()

                Fernet_ = Fernet(Key)

                #Key_name = 'MainKey.key'.format()
                shutil.copy2(os.path.join(Key_dir, Name_key), Key_path_chosen)

                # * This function sort the files and show them

                for Filename in os.listdir(Folder_path):

                    Filenames.append(Filename)

                    with open(Folder_path + '/' + Filename, 'rb') as File: # open in readonly mode
                        Original_file = File.read()
                    
                    Encrypted_File = Fernet_.encrypt(Original_file)

                    with open(Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                        Encrypted_file.write(Encrypted_File)

                with open(Key_path_chosen + '/' + Name_key + '.txt', "w") as text_file:
                    text_file.write('The key {} open the next documents {}'.format(Name_key, Filenames))  

            except OSError:
                print('Is not a key {} ❌'.format(str(Key_path))) #! Alert

# ? Decrypt files

@timer_func
def decrypt_files(**kwargs) -> None: 

    # * General parameters
    Folder_path = kwargs.get('folderpath', None)
    Key_path = kwargs.get('keypath', None)

    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Key_dir = os.path.dirname(Key_path)
    Key_file = os.path.basename(Key_path)

    Filename_key, Format = os.path.splitext(Key_file)

    Datetime = datetime.datetime.now()

    with open(Key_path, 'rb') as Filekey:
        Key = Filekey.read()

    Fernet_ = Fernet(Key)

    # * This function sort the files and show them

    if Filename_key.endswith('.key'):

        try:
            for Filename in os.listdir(Folder_path):

                print(Filename)

                with open(Folder_path + '/' + Filename, 'rb') as Encrypted_file: # open in readonly mode
                    Encrypted = Encrypted_file.read()
                
                Decrypted = Fernet_.decrypt(Encrypted)

                with open(Folder_path + '/' + Filename, 'wb') as Decrypted_file:
                    Decrypted_file.write(Decrypted)

            with open(Key_dir + '/' + Key_file + '.txt', "w") as text_file:
                    text_file.write('Key used. Datetime: {} '.format(Datetime))  

        except OSError:
                print('Is not a key {} ❌'.format(str(Key_path))) #! Alert

# ? Sort Files

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

# ? Remove all files in folder

@timer_func
def remove_all_files(Folder_path: str) -> None:
    """
    Remove all files inside the folder path obtained.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        None
    """
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # * This function will remove all the files inside a folder

    for File in os.listdir(Folder_path):
        Filename, Format  = os.path.splitext(File)
        print('Removing: {} . {} ✅'.format(Filename, Format))
        os.remove(os.path.join(Folder_path, File))

# ? Random remove all files in folder

@timer_func
def random_remove_files(Folder_path: str, Value: int) -> None:
    """
    Remove all files inside the folder path obtained.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        None
    """
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # * This function will remove all the files inside a folder
    Files = os.listdir(Folder_path)

        #Filename, Format = os.path.splitext(File)

    for File_sample in sample(Files, Value):
        print(File_sample)
        #print('Removing: {}{} ✅'.format(Filename, Format))
        os.remove(os.path.join(Folder_path, File_sample))

# ? ####################################################### Mini-MIAS #######################################################

# ? Extract the mean of each column

def extract_mean_from_images(Dataframe:pd.DataFrame, Column:int) -> int:
  """
  Extract the mean from the values of the whole dataset using its dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe with all the data needed(Mini-MIAS in this case).
      Column (int): The column number where it extracts the values.
  Returns:
      int: Return the mean from the column values.
  """

  # * This function will obtain the main of each column

  List_data_mean:list = []

  for i in range(Dataframe.shape[0]):
      if Dataframe.iloc[i - 1, Column] > 0:
          List_data_mean.append(Dataframe.iloc[i - 1, Column])

  Mean_list:int = int(np.mean(List_data_mean))
  return Mean_list
 
# ? Clean Mini-MIAS CSV

def mini_mias_csv_clean(Dataframe:pd.DataFrame) -> pd.DataFrame:
  """
  Clean the data from the Mini-MIAS dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe from Mini-MIAS' website.

  Returns:
      pd.DataFrame: Return the clean dataframe to use.
  """

  Value_fillna:int = 0
  # * This function will clean the data from the CSV archive

  Columns_list:list[str] = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
  Dataframe_Mini_MIAS = pd.read_csv(Dataframe, usecols = Columns_list)

  # * Severity's column
  Mini_MIAS_severity_column:int = 3

  # * it labels each severity grade

  LE = LabelEncoder()
  Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column].values
  Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column] = LE.fit_transform(Dataframe_Mini_MIAS.iloc[:, 3])

  # * Fullfill X, Y and RADIUS columns with 0
  Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].fillna(Value_fillna)
  Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].fillna(Value_fillna)
  Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].fillna(Value_fillna)

  #Dataframe["X"].replace({"*NOTE": 0}, inplace = True)
  #Dataframe["Y"].replace({"3*": 0}, inplace = True)

  # * X and Y columns tranform into int type
  Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].astype(int)
  Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].astype(int)

  # * Severity and radius columns tranform into int type
  Dataframe_Mini_MIAS['SEVERITY'] = Dataframe_Mini_MIAS['SEVERITY'].astype(int)
  Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].astype(int)

  return Dataframe_Mini_MIAS

# ?
@timer_func
class ChangeFormat:
  """
  _summary_

  _extended_summary_

  Raises:
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
  """
  # * Change the format of one image to another 

  def __init__(self, **kwargs):
    """
    _summary_

    _extended_summary_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
    """
    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.New_folder = kwargs.get('Newfolder', None)
    self.Format = kwargs.get('Format', None)
    self.New_format = kwargs.get('Newformat', None)

    # * Values, type errors.
    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.Folder, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    if self.New_folder == None:
      raise ValueError("Folder destination does not exist") #! Alert
    if not isinstance(self.New_folder, str):
      raise TypeError("Folder destination attribute must be a string") #! Alert

    if self.Format == None:
      raise ValueError("Current format does not exist") #! Alert
    if not isinstance(self.Format, str):
      raise TypeError("Current format must be a string") #! Alert

    if self.New_format == None:
      raise ValueError("New format does not exist") #! Alert
    if not isinstance(self.New_format, str):
      raise TypeError("Current format must be a string") #! Alert

  # * Folder attribute
  @property
  def Folder_property(self):
      return self.Folder

  @Folder_property.setter
  def Folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.Folder = New_value
  
  @Folder_property.deleter
  def Folder_property(self):
      print("Deleting folder...")
      del self.Folder

  # * New folder attribute
  @property
  def New_folder_property(self):
      return self.New_folder

  @New_folder_property.setter
  def New_folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.New_folder = New_value
  
  @New_folder_property.deleter
  def New_folder_property(self):
      print("Deleting folder...")
      del self.New_folder

  # * Format attribute
  @property
  def Format_property(self):
      return self.New_folder

  @Format_property.setter
  def Format_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.Format = New_value
  
  @Format_property.deleter
  def New_folder_property(self):
      print("Deleting folder...")
      del self.Format

  # * New Format attribute
  @property
  def New_format_property(self):
      return self.New_format

  @New_format_property.setter
  def New_format_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.New_format = New_value
  
  @New_format_property.deleter
  def New_format_property(self):
      print("Deleting folder...")
      del self.New_format

  def ChangeExtension(self):
    """
    _summary_

    _extended_summary_
    """
    # * Changes the current working directory to the given path
    os.chdir(self.Folder)
    print(os.getcwd())
    print("\n")

    # * Using the sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    Count:int = 0

    # * Reading the files
    for File in Sorted_files:
      if File.endswith(self.Format):

        try:
            Filename, Format  = os.path.splitext(File)
            print('Working with {} of {} {} images, {} ------- {} ✅'.format(Count, Total_images, self.Format, Filename, self.New_format))
            #print(f"Working with {Count} of {Total_images} {self.Format} images, {Filename} ------- {self.New_format} ✅")
            
            # * Reading each image using cv2
            Path_file = os.path.join(self.Folder, File)
            Image = cv2.imread(Path_file)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            # * Changing its format to a new one
            New_name_filename = Filename + self.New_format
            New_folder = os.path.join(self.New_folder, New_name_filename)

            cv2.imwrite(New_folder, Image)
            #FilenamesREFNUM.append(Filename)
            Count += 1

        except OSError:
            print('Cannot convert {} ❌'.format(str(File))) #! Alert
            #print('Cannot convert %s ❌' % File) #! Alert

    print("\n")
    #print(f"COMPLETE {Count} of {Total_images} TRANSFORMED ✅")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

# ? Class for images cropping.
@timer_func
class CropImages():
  """
  _summary_

  _extended_summary_
  """
  def __init__(self, **kwargs) -> None:
    
    """
    _summary_

    _extended_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    """
    # * This algorithm outputs crop values for images based on the coordinates of the CSV file.
    # * General parameters
    self.Folder: str = kwargs.get('Folder', None)
    self.Normalfolder: str = kwargs.get('Normalfolder', None)
    self.Tumorfolder: str = kwargs.get('Tumorfolder', None)
    self.Benignfolder: str = kwargs.get('Benignfolder', None)
    self.Malignantfolder: str = kwargs.get('Malignantfolder', None)

    # * CSV to extract data
    self.Dataframe: pd.DataFrame = kwargs.get('Dataframe', None)
    self.Shapes = kwargs.get('Shapes', None)
    
    # * X and Y mean to extract normal cropped images
    self.Xmean:int = kwargs.get('Xmean', None)
    self.Ymean:int = kwargs.get('Ymean', None)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Folder must be a string") #! Alert

    if self.Normalfolder == None:
      raise ValueError("Folder for normal images does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Folder must be a string") #! Alert

    if self.Tumorfolder == None:
      raise ValueError("Folder for tumor images does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Folder must be a string") #! Alert

    if self.Benignfolder == None:
      raise ValueError("Folder for benign images does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Folder must be a string") #! Alert

    if self.Malignantfolder == None:
      raise ValueError("Folder for malignant images does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Folder must be a string") #! Alert
      
    #elif self.Dataframe == None:
      #raise ValueError("The dataframe is required") #! Alert

    elif self.Shapes == None:
      raise ValueError("The shape is required") #! Alert

    elif self.Xmean == None:
      raise ValueError("Xmean is required") #! Alert

    elif self.Ymean == None:
      raise ValueError("Ymean is required") #! Alert

  def CropMIAS(self):
    
    #Images = []

    os.chdir(self.Folder)

    # * Columns
    Name_column = 0
    Severity = 3
    X_column = 4
    Y_column = 5
    Radius = 6

    # * Labels
    Benign = 0
    Malignant = 1
    Normal = 2

    # * Initial index
    Index = 1
    
    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
        Filename, Format = os.path.splitext(File)

        print("******************************************")
        print(self.Dataframe.iloc[Index - 1, Name_column])
        print(Filename)
        print("******************************************")

        if self.Dataframe.iloc[Index - 1, Severity] == Benign:
            if self.Dataframe.iloc[Index - 1, X_column] > 0  or self.Dataframe.iloc[Index - 1, Y_column] > 0:
              
                try:
                
                  print(f"Working with {Count} of {Total_images} {Format} Benign images, {Filename} X: {self.Dataframe.iloc[Index - 1, X_column]} Y: {self.Dataframe.iloc[Index - 1, Y_column]}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  # * Reading the image
                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)
                
                  #Distance = self.Shape # X and Y.
                  #Distance = self.Shape # Perimetro de X y Y de la imagen.
                  #Image_center = Distance / 2 
                    
                  # * Obtaining the center using the radius
                  Image_center = self.Dataframe.iloc[Index - 1, Radius] / 2 
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])
                  print(self.Dataframe.iloc[Index - 1, Radius])

                  # * Extract the value of X and Y of each image
                  X_size = self.Dataframe.iloc[Index - 1, X_column]
                  print(X_size)
                  Y_size = self.Dataframe.iloc[Index - 1, Y_column]
                  print(Y_size)
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  XDM = X_size + Image_center
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  YDM = Height_Y - Y_size + Image_center

                  # * Cropped image
                  Cropped_Image_Benig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Image.shape, " ----------> ", Cropped_Image_Benig.shape)

                  # print(Cropped_Image_Benig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)

                  New_name_filename = Filename + '_Benign_cropped' + Format

                  New_folder = os.path.join(self.Benignfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Benig)

                  New_folder = os.path.join(self.Tumorfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Benig)

                  #Images.append(Cropped_Image_Benig)

                except OSError:
                        print('Cannot convert %s' % File)

        elif self.Dataframe.iloc[Index - 1, Severity] == Malignant:
            if self.Dataframe.iloc[Index - 1, X_column] > 0  or self.Dataframe.iloc[Index - 1, Y_column] > 0:

                try:

                  print(f"Working with {Count} of {Total_images} {Format} Malignant images, {Filename} X {self.Dataframe.iloc[Index - 1, X_column]} Y {self.Dataframe.iloc[Index - 1, Y_column]}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  # * Reading the image
                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)
                
                  #Distance = self.Shape # X and Y.
                  #Distance = self.Shape # Perimetro de X y Y de la imagen.
                  #Image_center = Distance / 2 
                    
                  # * Obtaining the center using the radius
                  Image_center = self.Dataframe.iloc[Index - 1, Radius] / 2 # Center
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])

                  # * Extract the value of X and Y of each image
                  X_size = self.Dataframe.iloc[Index - 1, X_column]
                  Y_size = self.Dataframe.iloc[Index - 1, Y_column]
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  XDM = X_size + Image_center
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  YDM = Height_Y - Y_size + Image_center

                  # * Cropped image
                  Cropped_Image_Malig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Image.shape, " ----------> ", Cropped_Image_Malig.shape)
        
                  # print(Cropped_Image_Malig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)

                  New_name_filename = Filename + '_Malignant_cropped' + Format

                  New_folder = os.path.join(self.Malignantfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Malig)

                  New_folder = os.path.join(self.Tumorfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Malig)

                  #Images.append(Cropped_Image_Malig)


                except OSError:
                    print('Cannot convert %s' % File)
        
        elif self.Dataframe.iloc[Index - 1, Severity] == Normal:
          if self.Dataframe.iloc[Index - 1, X_column] == 0  or self.Dataframe.iloc[Index - 1, Y_column] == 0:

                try:

                  print(f"Working with {Count} of {Total_images} {Format} Normal images, {Filename}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)

                  Distance = self.Shapes # Perimetro de X y Y de la imagen.
                  Image_center = Distance / 2 # Centro de la imagen.
                  #CD = self.df.iloc[Index - 1, Radius] / 2
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])

                  # * Extract the value of X and Y of each image
                  X_size = self.Xmean
                  Y_size = self.Ymean
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  XDM = X_size + Image_center
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  YDM = Height_Y - Y_size + Image_center

                  # * Cropped image
                  Cropped_Image_Normal = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  # * Comparison two images
                  print(Image.shape, " ----------> ", Cropped_Image_Normal.shape)

                  # print(Cropped_Image_Normal.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)
              
                  New_name_filename = Filename + '_Normal_cropped' + Format

                  New_folder = os.path.join(self.Normalfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Normal)

                  #Images.append(Cropped_Image_Normal)

                except OSError:
                    print('Cannot convert %s' % File)

        Index += 1   

# ? Kmeans algorithm
@timer_func
def kmeans_function(Folder_CSV: str, Folder_graph: str, Technique_name: str, X_data, Clusters: int, Filename: str, Severity: str) -> pd.DataFrame:
  """
  _summary_

  _extended_summary_

  Args:
      Folder_CSV (str): _description_
      Folder_graph (str): _description_
      Technique_name (str): _description_
      X_data (_type_): _description_
      Clusters (int): _description_
      Filename (str): _description_
      Severity (str): _description_

  Returns:
      pd.DataFrame: _description_
  """

  # * Tuple with different colors
  List_wcss = []
  Colors = ('red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')

  for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_data)
    List_wcss.append(kmeans.inertia_)

  plt.plot(range(1, 10), List_wcss)
  plt.title('The Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  #plt.show()

  kmeans = KMeans(n_clusters = Clusters, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(X_data)

  for i in range(Clusters):

    if  Clusters <= 10:

        plt.scatter(X_data[y_kmeans == i, 0], X_data[y_kmeans == i, 1], s = 100, c = Colors[i], label = 'Cluster ' + str(i))


  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

  plt.title('Clusters')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend()

  # * Tuple with different colors
  Folder_graph_name = 'Kmeans_Graph_{}_{}.png'.format(Technique_name, Severity)
  Folder_graph_folder = os.path.join(Folder_graph, Folder_graph_name)
  plt.savefig(Folder_graph_folder)
  #plt.show()

  DataFrame = pd.DataFrame({'y_kmeans' : y_kmeans, 'REFNUM' : Filename})
  #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
  #print(DataFrame)

  #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
  Dataframe_name = '{}_Dataframe_{}'.format(Technique_name, Severity)
  Dataframe_folder = os.path.join(Folder_CSV, Dataframe_name)

  DataFrame.to_csv(Dataframe_folder)

  #print(DataFrame['y_kmeans'].value_counts())

  return DataFrame

  # ? Remove Data from K-means function
  
@timer_func
def kmeans_remove_data(Folder_path: str, Folder_CSV: str, Technique_name: str, Dataframe: pd.DataFrame, Cluster_to_remove: int, Severity: str) -> pd.DataFrame:
  """
  _summary_

  _extended_summary_

  Args:
      Folder_path (str): _description_
      Folder_CSV (str): _description_
      Technique_name (str): _description_
      Dataframe (pd.DataFrame): _description_
      Cluster_to_remove (int): _description_
      Severity (str): _description_

  Raises:
      ValueError: _description_

  Returns:
      pd.DataFrame: _description_
  """

  # * General lists
  #Images = [] # Png Images
  All_filename = [] 

  DataRemove = []
  Data = 0

  KmeansValue = 0
  Refnum = 1
  count = 1
  Index = 1

  os.chdir(Folder_path)

  # * Using sort function
  sorted_files, images = sort_images(Folder_path)

  # * Reading the files
  for File in sorted_files:

    Filename, Format = os.path.splitext(File)

    if Dataframe.iloc[Index - 1, Refnum] == Filename: # Read png files

      print(Filename)
      print(Dataframe.iloc[Index - 1, Refnum])

      if Dataframe.iloc[Index - 1, KmeansValue] == Cluster_to_remove:

        try:
          print(f"Working with {count} of {images} {Format} images, {Filename} ------- {Format} ✅")
          count += 1

          Path_File = os.path.join(Folder_path, File)
          os.remove(Path_File)
          print(Dataframe.iloc[Index - 1, Refnum], ' removed ❌')
          DataRemove.append(count)
          Data += 0

          #df = df.drop(df.index[count])

        except OSError:
          print('Cannot convert %s ❌' % File)

      elif Dataframe.iloc[Index - 1, KmeansValue] != Cluster_to_remove:
      
        All_filename.append(Filename)

      Index += 1

    elif Dataframe.iloc[Index - 1, Refnum] != Filename:
    
      print(Dataframe.iloc[Index - 1, Refnum]  + '----' + Filename)
      print(Dataframe.iloc[Index - 1, Refnum])
      raise ValueError("Files are not the same") #! Alert

    else:

      Index += 1

    for i in range(Data):

      Dataframe = Dataframe.drop(Dataframe.index[DataRemove[i]])

#Dataset = pd.DataFrame({'y_kmeans':df_u.iloc[Index - 1, REFNUM], 'REFNUM':df_u.iloc[Index - 1, KmeansValue]})
#X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

  #print(df)
  #pd.set_option('display.max_rows', df.shape[0] + 1)

  Dataframe_name = str(Technique_name) + '_Data_Removed_' + str(Severity) + '.csv'
  Dataframe_folder = os.path.join(Folder_CSV, Dataframe_name)

  Dataframe.to_csv(Dataframe_folder)

  return Dataframe

# ? ####################################################### CBIS-DDSM #######################################################

# ? CBIS-DDSM split data

def CBIS_DDSM_split_data(**kwargs) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV (_type_): _description_
        Folder (_type_): _description_
        Folder_total_benign (_type_): _description_
        Folder_benign (_type_): _description_
        Folder_benign_wc (_type_): _description_
        Folder_malignant (_type_): _description_
        Folder_abnormal (_type_): _description_
        Dataframe (_type_): _description_
        Severity (_type_): _description_
        Phase (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * General parameters
    Folder = kwargs.get('folder', None)
    Folder_total_benign = kwargs.get('Allbenignfolder', None)
    Folder_benign = kwargs.get('benignfolder', None)
    Folder_benign_wc = kwargs.get('benignWCfolder', None)
    Folder_malignant = kwargs.get('malignantfolder', None)
    Folder_abnormal = kwargs.get('Abnormalfolder', None)
    Folder_CSV = kwargs.get('csvfolder', None)

    Dataframe = kwargs.get('dataframe', None)
    Severity = kwargs.get('severity', None)
    Stage = kwargs.get('stage', None)

    Save_file = kwargs.get('savefile', False)

    # * Folder attribute (ValueError, TypeError)
    if Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder CSV (ValueError, TypeError)
    if Folder_CSV == None:
      raise ValueError("Folder to save csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
      raise TypeError("Folder to save csv must be a string") #! Alert

    # * Folder benign (ValueError, TypeError)
    if Folder_benign == None:
      raise ValueError("Folder benign does not exist") #! Alert
    if not isinstance(Folder_benign, str):
      raise TypeError("Folder benign must be a string") #! Alert

    # * Folder benign without callback (ValueError, TypeError)
    if Folder_benign_wc == None:
      raise ValueError("Folder benign without callback does not exist") #! Alert
    if not isinstance(Folder_benign_wc, str):
      raise TypeError("Folder abenign without callback must be a string") #! Alert

    # * Folder malignant (ValueError, TypeError)
    if Folder_malignant == None:
      raise ValueError("Folder malignant does not exist") #! Alert
    if not isinstance(Folder_malignant, str):
      raise TypeError("Folder malignant must be a string") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe == None:
      raise ValueError("Dataframe does not exist") #! Alert
    if not isinstance(Dataframe, pd.DataFrame):
      raise TypeError("Dataframe must be a dataframe") #! Alert

    # * Severity (ValueError, TypeError)
    if Severity == None:
      raise ValueError("Severity label does not exist") #! Alert
    if not isinstance(Severity, str):
      raise TypeError("Severity label must be a string") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Stage == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Save_file (ValueError, TypeError)
    if (Folder_CSV != None and Save_file == True):
      if not isinstance(Folder_CSV, str):
        raise TypeError("Folder destination must be a string") #! Alert
    elif (Folder_CSV == None and Save_file == True):
      warnings.warn('Saving the images is available but a folder destination was not found') #! Alert
      print("\n")
    elif (Folder_CSV != None and Save_file == False):
      warnings.warn('Saving the images is unavailable but a folder destination was found') #! Alert
      print("\n")
    else:
      pass

    #remove_all_files(Folder_total_benign)
    #remove_all_files(Folder_benign)
    #remove_all_files(Folder_benign_wc)
    #remove_all_files(Folder_malignant)
    #remove_all_files(Folder_abnormal)
    
    # * Lists to save the images and their respective labels
    Images = []
    Label = []

    # * Lists to save filename of each severity
    Filename_benign_all = []
    Filename_malignant_all = []
    Filename_all = []
    Filename_benign_list = []
    Filename_benign_WC_list = []
    Filename_malignant_list = []
    Filename_abnormal_list = []
    
    # * Label for each severity
    Benign_label = 0
    Benign_without_callback_label = 1
    Malignant_label = 2

    # * String label for each severity
    Benign_label_string = 'Benign'
    Benign_without_callback_label_string = 'Benign without callback'
    Malignant_label_string = 'Malignant'

    # * Initial index
    Index = 0
    
    # * Initial count variable
    Count = 1

    # * Change the current directory to specified directory
    os.chdir(Folder)

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder)

    for File in Sorted_files:
        
        # * Extract filename and format for each file
        Filename, Format = os.path.splitext(File)

        if Dataframe[Index] == Benign_label:

                try:
                    
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_benign = cv2.imread(File_folder)
                    
                    # * Combine name with the format
                    #Filename_benign = Filename + '_Benign'
                    Filename_benign = '{}_Benign'.format(str(Filename))
                    Filename_benign_format = Filename_benign + Format
                    
                    # * Save images in their folder, respectively
                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign)

                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_benign)

                    Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign)

                    # * Add data into the lists
                    Images.append(Image_benign)
                    Label.append(Benign_label)

                    Filename_benign_all.append(Filename_benign)
                    Filename_abnormal_list.append(Filename_benign)
                    Filename_benign_list.append(Filename_benign)
                    Filename_all.append(Filename_benign)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe[Index] == Benign_without_callback_label:
    
                try:

                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")
                    
                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_benign_without_callback = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_benign_WC = Filename + '_Benign_Without_Callback'
                    Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                    Filename_benign_WC_format = Filename_benign_WC + Format

                    # * Save images in their folder, respectively
                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign_without_callback)

                    Filename_benign_folder = os.path.join(Folder_benign_wc, Filename_benign_WC_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign_without_callback)

                    # * Add data into the lists
                    Images.append(Image_benign_without_callback)
                    Label.append(Benign_without_callback_label)

                    Filename_benign_all.append(Filename_benign_WC)
                    Filename_benign_WC_list.append(Filename_benign_WC)
                    Filename_all.append(Filename_benign_WC)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe[Index] == Malignant_label:

                try:

                    print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string ), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
                    
        Index += 1

    Dataframe_labeled = pd.DataFrame({'Filenames':Filename_all,'Labels':Label}) 

    if Save_file == True:
        #Dataframe_labeled_name = 'CBIS_DDSM_Split_' + 'Dataframe_' + str(Severity) + '_' + str(Stage) + '.csv' 
        Dataframe_labeled_name = 'CBIS_DDSM_Split_Dataframe_{}_{}.csv'.format(str(Severity), str(Stage))
        Dataframe_labeled_folder = os.path.join(Folder_CSV, Dataframe_labeled_name)
        Dataframe_labeled.to_csv(Dataframe_labeled_folder)

    return Dataframe_labeled

# ? CBIS-DDSM split all data

def CBIS_DDSM_split_several_data(**kwargs)-> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV (_type_): _description_
        Folder_test (_type_): _description_
        Folder_training (_type_): _description_
        Folder_total_benign (_type_): _description_
        Folder_benign (_type_): _description_
        Folder_benign_wc (_type_): _description_
        Folder_malignant (_type_): _description_
        Folder_abnormal (_type_): _description_
        Dataframe_test (_type_): _description_
        Dataframe_training (_type_): _description_
        Severity (_type_): _description_
        Phase (_type_): _description_

    Returns:
        _type_: _description_
    """

    # * General parameters
    Folder_test = kwargs.get('testfolder', None)
    Folder_training = kwargs.get('trainingfolder', None)
    Folder_total_benign = kwargs.get('Allbenignfolder', None)
    Folder_benign = kwargs.get('benignfolder', None)
    Folder_benign_wc = kwargs.get('benignWCfolder', None)
    Folder_malignant = kwargs.get('malignantfolder', None)
    Folder_abnormal = kwargs.get('Abnormalfolder', None)
    Folder_CSV = kwargs.get('csvfolder', None)

    Dataframe_test = kwargs.get('dftest', None)
    Dataframe_training = kwargs.get('dftraining', None)
    Severity = kwargs.get('severity', None)
    Stage_test = kwargs.get('stage', None)
    Stage_training = kwargs.get('stage', None)

    Save_file = kwargs.get('savefile', False)

    remove_all_files(Folder_total_benign)
    remove_all_files(Folder_benign)
    remove_all_files(Folder_benign_wc)
    remove_all_files(Folder_malignant)
    remove_all_files(Folder_abnormal)

    # * Folder test (ValueError, TypeError)
    if Folder_test == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_test, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder training (ValueError, TypeError)
    if Folder_training == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_training, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder CSV (ValueError, TypeError)
    if Folder_CSV == None:
      raise ValueError("Folder to save csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
      raise TypeError("Folder to save csv must be a string") #! Alert

    # * Folder benign (ValueError, TypeError)
    if Folder_benign == None:
      raise ValueError("Folder benign does not exist") #! Alert
    if not isinstance(Folder_benign, str):
      raise TypeError("Folder benign must be a string") #! Alert

    # * Folder benign without callback (ValueError, TypeError)
    if Folder_benign_wc == None:
      raise ValueError("Folder benign without callback does not exist") #! Alert
    if not isinstance(Folder_benign_wc, str):
      raise TypeError("Folder abenign without callback must be a string") #! Alert

    # * Folder malignant (ValueError, TypeError)
    if Folder_malignant == None:
      raise ValueError("Folder malignant does not exist") #! Alert
    if not isinstance(Folder_malignant, str):
      raise TypeError("Folder malignant must be a string") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe_test == None:
      raise ValueError("Dataframe test does not exist") #! Alert
    if not isinstance(Dataframe_test, pd.DataFrame):
      raise TypeError("Dataframe test must be a dataframe") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe_training == None:
      raise ValueError("Dataframe training does not exist") #! Alert
    if not isinstance(Dataframe_training, pd.DataFrame):
      raise TypeError("Dataframe training must be a dataframe") #! Alert

    # * Severity (ValueError, TypeError)
    if Severity == None:
      raise ValueError("Severity label does not exist") #! Alert
    if not isinstance(Severity, str):
      raise TypeError("Severity label must be a string") #! Alert

    # * Stage test (ValueError, TypeError)
    if Stage_test == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage_test, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Stage training (ValueError, TypeError)
    if Stage_test == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage_test, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Save_file (ValueError, TypeError)
    if (Folder_CSV != None and Save_file == True):
      if not isinstance(Folder_CSV, str):
        raise TypeError("Folder destination must be a string") #! Alert
    elif (Folder_CSV == None and Save_file == True):
      warnings.warn('Saving the images is available but a folder destination was not found') #! Alert
      print("\n")
    elif (Folder_CSV != None and Save_file == False):
      warnings.warn('Saving the images is unavailable but a folder destination was found') #! Alert
      print("\n")
    else:
      pass

    # * Lists to save the images and their respective labels
    Images = []
    Label = []

    # * Lists to save filename of each severity
    Filename_benign_all = []
    Filename_malignant_all = []
    Filename_all = []

    Filename_benign_list = []
    Filename_benign_WC_list = []
    Filename_malignant_list = []
    Filename_abnormal_list = []
    
    # * Label for each severity
    Benign_label = 0
    Benign_without_callback_label = 1
    Malignant_label = 2

    # * String label for each severity
    Benign_label_string = 'Benign'
    Benign_without_callback_label_string = 'Benign without callback'
    Malignant_label_string = 'Malignant'

    # * Initial index
    Index = 0
    
    # * Initial count variable
    Count = 1

    # * Change the current directory to specified directory - Test
    os.chdir(Folder_test)

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder_test)
    
    for File in Sorted_files:
        
        Filename, Format  = os.path.splitext(File)
        
        if Dataframe_test[Index] == Benign_label:

            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign = '{}_Benign'.format(str(Filename))
                Filename_benign_format = Filename_benign + Format
                
                # * Save images in their folder, respectively
                Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                cv2.imwrite(Filename_abnormal_folder, Image_benign)

                Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                cv2.imwrite(Filename_total_benign_folder, Image_benign)

                Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                cv2.imwrite(Filename_benign_folder, Image_benign)

                # * Add data into the lists
                Images.append(Image_benign)
                Label.append(Benign_label)

                Filename_benign_all.append(Filename_benign)
                Filename_abnormal_list.append(Filename_benign)
                Filename_benign_list.append(Filename_benign)
                Filename_all.append(Filename_benign)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe_test[Index] == Benign_without_callback_label:
    
            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign_without_callback = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                Filename_benign_WC_format = Filename_benign_WC + Format
                
                # * Save images in their folder, respectively
                Filename_total_benign_WC_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_total_benign_WC_folder, Image_benign_without_callback)

                Filename_benign_WC_folder = os.path.join(Folder_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_benign_WC_folder, Image_benign_without_callback)

                # * Add data into the lists
                Images.append(Image_benign_without_callback)
                Label.append(Benign_without_callback_label)

                Filename_benign_all.append(Filename_benign_WC)
                Filename_benign_WC_list.append(Filename_benign_WC)
                Filename_all.append(Filename_benign_WC)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe_test[Index] == Malignant_label:

                try:
                    #print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string), str(Filename)))

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder_test, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        Index += 1
        #print(Index)
    
    # * Change the current directory to specified directory - Training
    os.chdir(Folder_training)
    
    # * Change the variable's value
    Index = 0
    count = 1

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder_training)
    
    for File in Sorted_files:
        
        Filename, Format  = os.path.splitext(File)
        
        if Dataframe_test[Index] == Benign_label:

            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign = '{}_Benign'.format(str(Filename))
                Filename_benign_format = Filename_benign + Format
                
                # * Save images in their folder, respectively
                Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                cv2.imwrite(Filename_abnormal_folder, Image_benign)

                Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                cv2.imwrite(Filename_total_benign_folder, Image_benign)

                Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                cv2.imwrite(Filename_benign_folder, Image_benign)

                # * Add data into the lists
                Images.append(Image_benign)
                Label.append(Benign_label)

                Filename_benign_all.append(Filename_benign)
                Filename_abnormal_list.append(Filename_benign)
                Filename_benign_list.append(Filename_benign)
                Filename_all.append(Filename_benign)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe_test[Index] == Benign_without_callback_label:
    
            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign_without_callback = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                Filename_benign_WC_format = Filename_benign_WC + Format
                
                # * Save images in their folder, respectively
                Filename_total_benign_WC_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_total_benign_WC_folder, Image_benign_without_callback)

                Filename_benign_WC_folder = os.path.join(Folder_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_benign_WC_folder, Image_benign_without_callback)

                # * Add data into the lists
                Images.append(Image_benign_without_callback)
                Label.append(Benign_without_callback_label)

                Filename_benign_all.append(Filename_benign_WC)
                Filename_benign_WC_list.append(Filename_benign_WC)
                Filename_all.append(Filename_benign_WC)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe_test[Index] == Malignant_label:

                try:
                    #print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string), str(Filename)))

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder_test, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        Index += 1

    Dataframe_labeled = pd.DataFrame({'Filenames':Filename_all,'Labels':Label}) 

    if Save_file == True:
        #Dataframe_labeled_name = 'CBIS_DDSM_Split_' + 'Dataframe_' + str(Severity) + '_' + str(Stage) + '.csv' 
        Dataframe_labeled_name = 'CBIS_DDSM_Split_Dataframe_{}_{}_{}.csv'.format(str(Severity), str(Stage_test), str(Stage_training))
        Dataframe_labeled_folder = os.path.join(Folder_CSV, Dataframe_labeled_name)
        Dataframe_labeled.to_csv(Dataframe_labeled_folder)

    return Dataframe_labeled

# ? Dataset splitting

def Dataframe_split(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
  """
  _summary_

  _extended_summary_

  Args:
      Dataframe (pd.DataFrame): _description_

  Returns:
      tuple[pd.DataFrame, set, set, set]: _description_
  """
  X = Dataframe.drop('Labels', axis = 1)
  Y = Dataframe['Labels']

  Majority, Minority = Dataframe['Labels'].value_counts()

  return X, Y, Majority, Minority

# ? Imbalance data majority

def Imbalance_data_majority(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
    """
    _summary_

    _extended_summary_

    Args:
        Dataframe (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, set, set, set]: _description_
    """
    X = Dataframe.drop('Labels', axis = 1)
    Y = Dataframe['Labels']

    # * Value counts from Y(Labels)
    Majority, Minority = Y.value_counts()

    Dataframe_majority = Dataframe[Y == 0]
    Dataframe_minority = Dataframe[Y == 1]
    
    # * Resample using majority
    Dataframe_majority_downsampled = resample(  Dataframe_majority, 
                                                replace = False,        # sample with replacement
                                                n_samples = Minority,   # to match majority class
                                                random_state = 123)     # reproducible results
    
    # * Concat minority and majority downsampled
    Dataframe_downsampled = pd.concat([Dataframe_minority, Dataframe_majority_downsampled])
    print(Dataframe_downsampled['Labels'].value_counts())

    X = Dataframe_downsampled.drop('Labels', axis = 1)
    Y = Dataframe_downsampled['Labels']

    return X, Y, Majority, Minority

# ? Imbalance data minority

def Imbalance_data_minority(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
    """
    _summary_

    _extended_summary_

    Args:
        Dataframe (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, set, set, set]: _description_
    """
    X = Dataframe.drop('Labels', axis = 1)
    Y = Dataframe['Labels']
    
    # * Value counts from Y(Labels)
    Majority, Minority = Y.value_counts()

    Dataframe_majority = Dataframe[Y == 0]
    Dataframe_minority = Dataframe[Y == 1]

    # * Resample using minority
    Dataframe_minority_upsampled = resample(    Dataframe_minority, 
                                                replace = True,         # sample with replacement
                                                n_samples = Majority,   # to match majority class
                                                random_state = 123)     # reproducible results

    # * Concat majority and minority upsampled
    Dataframe_upsampled = pd.concat([Dataframe_majority, Dataframe_minority_upsampled])
    print(Dataframe_upsampled['Labels'].value_counts())

    X = Dataframe_upsampled.drop('Labels', axis = 1)
    Y = Dataframe_upsampled['Labels']

    return X, Y, Majority, Minority

# ? Convertion severity to int value

def CBIS_DDSM_CSV_severity_labeled(Folder_CSV: str, Column: int, Severity: int)-> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV(str): _description_
        Column(int): _description_
        Severity(int): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # * Folder attribute (ValueError, TypeError)
    if Folder_CSV == None:
        raise ValueError("Folder csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
        raise TypeError("Folder csv must be a string") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Column == None:
        raise ValueError("Column does not exist") #! Alert
    if not isinstance(Column, int):
        raise TypeError("Column must be a integer") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Severity == None:
        raise ValueError("Severity does not exist") #! Alert
    if Severity >= 3:
        raise ValueError("Severity must be less than 3") #! Alert
    if not isinstance(Severity, int):
        raise TypeError("Severity must be a integer") #! Alert

    # * Label encoder class
    LE = LabelEncoder()

    # * Severity labels
    Calcification = 1
    Mass = 2

    # * Dataframe headers
    if Severity == Calcification:

        Columns_list = ["patient_id", "breast density", "left or right breast", "image view", 
                        "abnormality id", "abnormality type", "calc type", "calc distribution", 
                        "assessment", "pathology", "subtlety", "image file path", "cropped image file path", 
                        "ROI mask file path"]
    if Severity == Mass:

        Columns_list = ["patient_id", "breast_density", "left or right breast", "image view", 
                        "abnormality id", "abnormality type", "mass shape", "mass margins", 
                        "assessment", "pathology", "subtlety", "image file path", "cropped image file path", 
                        "ROI mask file path"]
    
    # * Dataframe headers between calfications or masses
    Dataframe_severity = pd.read_csv(Folder_CSV, usecols = Columns_list)

    # * Getting the values and label them
    Dataframe_severity.iloc[:, Column].values
    Dataframe_severity.iloc[:, Column] = LE.fit_transform(Dataframe_severity.iloc[:, Column])

    Dataset_severity_labeled = Dataframe_severity.iloc[:, Column].values
    Dataframe = Dataframe_severity.iloc[:, Column]

    print(Dataset_severity_labeled)
    pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
    print(Dataframe.value_counts())

    return Dataset_severity_labeled

# ? Concat multiple dataframes
@timer_func
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

# ? Split folders into train/test/validation
@timer_func
def split_folders_train_test_val(Folder_path:str, Only_train_test: bool) -> str:
  """
  Create a new folder with the folders of the class problem and its distribution of training, test and validation.
  The split is 80 and 20. If there is a validation set, it'll be 80, 10, and 10.

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

  New_Folder_name = '{}_Split'.format(Folder_path)

  print("*" * Asterisks)
  print('New folder name: {}'.format(New_Folder_name))
  print("*" * Asterisks)

  #1337
  
  try:

    if(Only_train_test == False):

      splitfolders.ratio(Folder_path, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 
    
    else:

      Test_split: float = 0.2
      splitfolders.ratio(Folder_path, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split)) 

  except OSError as e:
    print('Cannot split the following folder {}, Type error: {} ❌'.format(str(Folder_path), str(type(e)))) #! Alert

  return New_Folder_name

# ? .
@timer_func
class DCM_format():

  def __init__(self, **kwargs:string) -> None:
    
    # * This algorithm outputs crop values for images based on the coordinates of the CSV file.

    # * Instance attributes folders
    self.Folder = kwargs.get('folder', None)
    self.Folder_all = kwargs.get('allfolder', None)
    self.Folder_patches = kwargs.get('patchesfolder', None)
    self.Folder_resize = kwargs.get('resizefolder', None)
    self.Folder_resize_normalize = kwargs.get('normalizefolder', None)

    # * Instance attributes labels
    self.Severity = kwargs.get('Severity', None)
    self.Stage = kwargs.get('Phase', None)

    # * Folder attribute (ValueError, TypeError)
    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.Folder, str):
      raise TypeError("Folder must be a string") #! Alert

    # * Folder destination where all the new images will be stored (ValueError, TypeError)
    if self.Folder_all == None:
      raise ValueError("Destination folder does not exist") #! Alert
    if not isinstance(self.Folder_all, str):
      raise TypeError("Destination folder must be a string") #! Alert

    # * Folder normal to stored images without preprocessing from CBIS-DDSM (ValueError, TypeError)
    if self.Folder_patches == None:
      raise ValueError("Normal folder does not exist") #! Alert
    if not isinstance(self.Folder_patches, str):
      raise TypeError("Normal folder must be a string") #! Alert

    # * Folder normal to stored resize images from CBIS-DDSM (ValueError, TypeError)
    if self.Folder_resize == None:
      raise ValueError("Resize folder does not exist") #! Alert
    if not isinstance(self.Folder_resize, str):
      raise TypeError("Resize folder must be a string") #! Alert

    # * Folder normal to stored resize normalize images from CBIS-DDSM (ValueError, TypeError)
    if self.Folder_resize_normalize == None:
      raise ValueError("Normalize resize folder images does not exist") #! Alert
    if not isinstance(self.Folder_resize_normalize, str):
      raise TypeError("Normalize resize folder must be a string") #! Alert

    # * Severity label (ValueError, TypeError)
    if self.Severity == None:
      raise ValueError("Severity does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Severity must be a string") #! Alert

    # * Phase label (ValueError, TypeError)
    if self.Stage == None:
      raise ValueError("Phase images does not exist") #! Alert
    if not isinstance(self.Stage, str):
      raise TypeError("Phase must be a string") #! Alert

  def __repr__(self) -> str:

        kwargs_info = "Folder: {} , Folder_all: {}, Folder_normal: {}, Folder_resize: {}, Folder_resize_normalize: {}, Severity: {}, Phase: {}".format( self.Folder, 
                                                                                                                                                        self.Folder_all, self.Folder_patches,
                                                                                                                                                        self.Folder_resize, self.Folder_resize_normalize, 
                                                                                                                                                        self.Severity, self.Stage )
        return kwargs_info

  def __str__(self) -> str:

        Descripcion_class = ""
        
        return Descripcion_class

  # * Folder attribute
  @property
  def Folder_property(self):
      return self.Folder

  @Folder_property.setter
  def Folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.Folder = New_value
  
  @Folder_property.deleter
  def Folder_property(self):
      print("Deleting folder...")
      del self.Folder

  # * Folder all images attribute
  @property
  def Folder_all_property(self):
      return self.Folder_all

  @Folder_all_property.setter
  def Folder_all_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder all must be a string") #! Alert
      self.Folder_all = New_value
  
  @Folder_all_property.deleter
  def Folder_all_property(self):
      print("Deleting all folder...")
      del self.Folder_all

  # * Folder patches images attribute
  @property
  def Folder_patches_property(self):
      return self.Folder_patches

  @Folder_patches_property.setter
  def Folder_patches_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder patches must be a string") #! Alert
      self.Folder_patches = New_value
  
  @Folder_patches_property.deleter
  def Folder_patches_property(self):
      print("Deleting patches folder...")
      del self.Folder_patches

  # * Folder resize images attribute
  @property
  def Folder_resize_property(self):
      return self.Folder_resize

  @Folder_resize_property.setter
  def Folder_resize_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder resize must be a string") #! Alert
      self.Folder_resize = New_value
  
  @Folder_resize_property.deleter
  def Folder_resize_property(self):
      print("Deleting resize folder...")
      del self.Folder_resize

  # * Folder resize normalize images attribute
  @property
  def Folder_resize_normalize_property(self):
      return self.Folder_resize_normalize

  @Folder_resize_normalize_property.setter
  def Folder_resize_normalize_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder resize normalize must be a string") #! Alert
      self.Folder_resize_normalize = New_value
  
  @Folder_resize_normalize_property.deleter
  def Folder_resize_normalize_property(self):
      print("Deleting resize normalize folder...")
      del self.Folder_resize_normalize

  # * Severity attribute
  @property
  def Severity_property(self):
      return self.Severity

  @Severity_property.setter
  def Severity_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Severity must be a string") #! Alert
      self.Severity = New_value
  
  @Severity_property.deleter
  def Severity_property(self):
      print("Deleting severity...")
      del self.Severity
  
  # * Stage
  @property
  def Stage_property(self):
      return self.Stage

  @Stage_property.setter
  def Stage_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Stage must be a string") #! Alert
      self.Stage = New_value
  
  @Stage_property.deleter
  def Stage_property(self):
      print("Deleting stage...")
      del self.Stage

  def DCM_change_format(self) -> None:
    """
    Printing amount of images with data augmentation

    Args:
        Folder_path (str): Folder's dataset for distribution

    Returns:
        None
    """

    # * Format DCM and PNG variables
    DCM = ".dcm"
    PNG = ".png"

    # * Initial file
    File = 0

    # * Standard parameters for resize
    X_size_resize = 224
    Y_size_resize = 224

    # * General lists and string DCM
    DCM_files = []
    DCM_files_sizes = []
    DCM_Filenames = []

    # * Interpolation that is used
    Interpolation = cv2.INTER_CUBIC

    # * Shape for the resize
    Shape_resize = (X_size_resize, Y_size_resize)

    # * Read images from folder
    Files_total = os.listdir(self.Folder)

    # * Sorted files and multiply them
    Files_total_ = Files_total * 2
    Files_total_ = sorted(Files_total_)

    # * Search for each dir and file inside the folder given
    for Root, Dirs, Files in os.walk(self.Folder, True):
        print("root:%s"% Root)
        print("dirs:%s"% Dirs)
        print("files:%s"% Files)
        print("-------------------------------")

    for Root, Dirs, Files in os.walk(self.Folder):
        for x in Files:
            if x.endswith(DCM):
                DCM_files.append(os.path.join(Root, x))

    # * Sorted DCM files
    DCM_files = sorted(DCM_files)
    
    # * Get the size of each dcm file
    for i in range(len(DCM_files)):
        DCM_files_sizes.append(os.path.getsize(DCM_files[i]))

    # * put it together in a dataframe
    DCM_dataframe_files = pd.DataFrame({'Path':DCM_files, 'Size':DCM_files_sizes, 'Filename':Files_total_}) 
    print(DCM_dataframe_files)

    Total_DCM_files = len(DCM_files_sizes)

    # * Search inside each folder to get the archive which has the less size.
    for i in range(0, Total_DCM_files, 2):

        print(DCM_files_sizes[i], '----', DCM_files_sizes[i + 1])

        if DCM_files_sizes[i] > DCM_files_sizes[i + 1]:
            DCM_dataframe_files.drop([i], axis = 0, inplace = True)
        else:
            DCM_dataframe_files.drop([i + 1], axis = 0, inplace = True)

    # * Several prints
    print(len(DCM_files))
    print(len(DCM_files_sizes))
    print(len(Files_total_))
    print(DCM_dataframe_files)

    # * Get the columns of DCM filenames
    DCM_filenames = DCM_dataframe_files.iloc[:, 0].values
    Total_DCM_filenames = DCM_dataframe_files.iloc[:, 2].values

    # * Write the dataframe in a folder
    #DCM_dataframe_name = 'DCM_' + 'Format_' + str(self.Severity) + '_' + str(self.Phase) + '.csv'
    DCM_dataframe_name = 'DCM_Format_{}_{}.csv'.format(str(self.Severity), str(self.Stage))
    DCM_dataframe_folder = os.path.join(self.Folder_all, DCM_dataframe_name)
    DCM_dataframe_files.to_csv(DCM_dataframe_folder)

    # * Convert each image from DCM format to PNG format
    for File in range(len(DCM_dataframe_files)):

        # * Read DCM format using pydicom
        DCM_read_pydicom_file = pydicom.dcmread(DCM_Filenames[File])
        
        # * Convert to float type
        DCM_image = DCM_read_pydicom_file.pixel_array.astype(float)

        # * Rescaled and covert to float64
        DCM_image_rescaled = (np.maximum(DCM_image, 0) / DCM_image.max()) * 255.0
        DCM_image_rescaled_float64 = np.float64(DCM_image_rescaled)

        # * Get a new images to the normalize(zeros)
        DCM_black_image = np.zeros((X_size_resize, Y_size_resize))

        # * Use the resize function
        DCM_image_resize = cv2.resize(DCM_image_rescaled_float64, Shape_resize, interpolation = Interpolation)

        # * Use the normalize function with the resize images
        DCM_image_normalize = cv2.normalize(DCM_image_resize, DCM_black_image, 0, 255, cv2.NORM_MINMAX)

        # * Get each image and convert them
        DCM_file = Total_DCM_filenames[File]
        DCM_name_file = '{}{}'.format(str(DCM_file), str(PNG))
        #DCM_name_file = str(DCM_file) + '.png'

        # * Save each transformation in different folders
        DCM_folder = os.path.join(self.Folder_patches, DCM_name_file)
        DCM_folder_resize = os.path.join(self.Folder_resize, DCM_name_file)
        DCM_folder_normalize = os.path.join(self.Folder_resize_normalize, DCM_name_file)

        cv2.imwrite(DCM_folder, DCM_image_rescaled_float64)
        cv2.imwrite(DCM_folder_resize, DCM_image_resize)
        cv2.imwrite(DCM_folder_normalize, DCM_image_normalize)

        # * Print for comparison
        print('Images: ', DCM_Filenames[File], '------', Total_DCM_filenames[File])    

# ?

class FigureAdjust():
  
  def __init__(self, **kwargs) -> None:

    # *
    self.Folder_path = kwargs.get('folder', None)
    self.Title = kwargs.get('title', None)

    # * 
    self.Show_image = kwargs.get('SI', False)
    self.Save_figure = kwargs.get('SF', False)

    # *
    self.Num_classes = kwargs.get('classes', None)

    # *
    self.X_figure_size = 12
    self.Y_figure_size = 12

    # * General parameters
    self.Font_size_title = self.X_figure_size * 1.2
    self.Font_size_general = self.X_figure_size * 0.8
    self.Font_size_ticks = (self.X_figure_size * self.Y_figure_size) * 0.05

    # * 
    #self.Annot_kws = kwargs.get('annot_kws', None)
    #self.Font = kwargs.get('font', None)
  
  def __repr__(self) -> str:

        kwargs_info = ''

        return kwargs_info

  def __str__(self) -> str:

        Descripcion_class = ''
        
        return Descripcion_class

  # * Folder_path attribute
  @property
  def Folder_path_property(self):
      return self.Folder_path

  @Folder_path_property.setter
  def Folder_path_property(self, New_value):
      self.Folder_path = New_value
  
  @Folder_path_property.deleter
  def Folder_path_property(self):
      print("Deleting Folder_path...")
      del self.Folder_path

  # * Title attribute
  @property
  def Title_property(self):
      return self.Title

  @Title_property.setter
  def Title_property(self, New_value):
      self.Title = New_value
  
  @Title_property.deleter
  def Title_property(self):
      print("Deleting Title...")
      del self.Title

  # * Show_image attribute
  @property
  def Show_image_property(self):
      return self.Show_image

  @Show_image_property.setter
  def Show_image_property(self, New_value):
      self.Show_image = New_value
  
  @Show_image_property.deleter
  def Show_image_property(self):
      print("Deleting Show_image...")
      del self.Show_image

  # * Save_figure attribute
  @property
  def Save_figure_property(self):
      return self.Save_figure

  @Save_figure_property.setter
  def Save_figure_property(self, New_value):
      self.Save_figure = New_value
  
  @Save_figure_property.deleter
  def Save_figure_property(self):
      print("Deleting Save_figure...")
      del self.Save_figure

  # * Num_classes attribute
  @property
  def Num_classes_property(self):
      return self.Num_classes

  @Num_classes_property.setter
  def Num_classes_property(self, New_value):
      self.Num_classes = New_value
  
  @Num_classes_property.deleter
  def Num_classes_property(self):
      print("Deleting Num_classes...")
      del self.Num_classes

  # ? Decorator
  @staticmethod
  def show_figure(Show_image: bool = False) -> None:

    if(Show_image == True):
      plt.show()
    
    else: 
      pass

  # ? Decorator
  @staticmethod
  def save_figure(Save_figure: bool, Title: int, Func_: str, Folder: str) -> None:

      if(Save_figure == True):
        
        Figure_name = 'Figure_{}_{}.png'.format(Title, Func_)
        Figure_folder = os.path.join(Folder, Figure_name)
        plt.savefig(Figure_folder)

      else:
        pass
    
# ?
class BarChart(FigureAdjust):
  """
  _summary_

  _extended_summary_
  """
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

    # *
    self.CSV_path = kwargs.get('csv', None)

    # *
    self.Plot_x_label = kwargs.get('label', None)
    self.Plot_column = kwargs.get('column', None)
    self.Plot_reverse = kwargs.get('reverse', None)

    # * Read dataframe csv
    self.Dataframe = pd.read_csv(self.CSV_path)

    # *
    self.Colors = ('gray', 'red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')
    
    # * General lists
    self.X_fast_list_values = []
    self.X_slow_list_values = []

    self.Y_fast_list_values = []
    self.Y_slow_list_values = []

    self.X_fastest_list_value = []
    self.Y_fastest_list_value = []

    self.X_slowest_list_value = []
    self.Y_slowest_list_value = []

    # * Chosing label
    if self.Num_classes == 2:
      self.Label_class_name = 'Biclass'
    elif self.Num_classes > 2:
      self.Label_class_name = 'Multiclass'

  # * CSV_path attribute
  @property
  def CSV_path_property(self):
      return self.CSV_path

  @CSV_path_property.setter
  def CSV_path_property(self, New_value):
      self.CSV_path = New_value
  
  @CSV_path_property.deleter
  def CSV_path_property(self):
      print("Deleting CSV_path...")
      del self.CSV_path

  # * Plot_x_label attribute
  @property
  def Plot_x_label_property(self):
      return self.Plot_x_label

  @Plot_x_label_property.setter
  def Plot_x_label_property(self, New_value):
      self.Plot_x_label = New_value
  
  @Plot_x_label_property.deleter
  def Plot_x_label_property(self):
      print("Deleting Plot_x_label...")
      del self.Plot_x_label

  # * Plot_column attribute
  @property
  def Plot_column_property(self):
      return self.Plot_column

  @Plot_column_property.setter
  def Plot_column_property(self, New_value):
      self.Plot_column = New_value
  
  @Plot_column_property.deleter
  def Plot_column_property(self):
      print("Deleting Plot_column...")
      del self.Plot_column

  # * Plot_reverse attribute
  @property
  def Plot_reverse_property(self):
      return self.Plot_reverse

  @Plot_reverse_property.setter
  def Plot_reverse_property(self, New_value):
      self.Plot_reverse = New_value
  
  @Plot_reverse_property.deleter
  def Plot_reverse_property(self):
      print("Deleting Plot_reverse...")
      del self.Plot_reverse

  # * Name attribute
  @property
  def Name_property(self):
      return self.Name

  @Name_property.setter
  def Name_property(self, New_value):
      self.Name = New_value
  
  @Name_property.deleter
  def Name_property(self):
      print("Deleting Name...")
      del self.Name

  @staticmethod  # no default first argument in logger function
  def barchart(func):  # accepts a function
      @wraps(func)  # good practice https://docs.python.org/2/library/functools.html#functools.wraps
      def wrapper(self, *args, **kwargs):  # explicit self, which means this decorator better be used inside classes only

          # * Get X and Y values
          X = list(self.Dataframe.iloc[:, 1])
          Y = list(self.Dataframe.iloc[:, self.Plot_column])

          plt.figure(figsize = (self.X_figure_size, self.Y_figure_size))

          # * Reverse is a bool variable with the postion of the plot
          if self.Plot_reverse == True:

              for Index, (i, k) in enumerate(zip(X, Y)):
                  if k < np.mean(Y):
                      self.X_fast_list_values.append(i)
                      self.Y_fast_list_values.append(k)
                  elif k >= np.mean(Y):
                      self.X_slow_list_values.append(i)
                      self.Y_slow_list_values.append(k)

              for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
                  if k == np.min(self.Y_fast_list_values):
                      self.X_fastest_list_value.append(i)
                      self.Y_fastest_list_value.append(k)
                      #print(X_fastest_list_value)
                      #print(Y_fastest_list_value)

              for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
                  if k == np.max(self.Y_slow_list_values):
                      self.X_slowest_list_value.append(i)
                      self.Y_slowest_list_value.append(k)
          else:
              for Index, (i, k) in enumerate(zip(X, Y)):
                  if k < np.mean(Y):
                      self.X_slow_list_values.append(i)
                      self.Y_slow_list_values.append(k)
                  elif k >= np.mean(Y):
                      self.X_fast_list_values.append(i)
                      self.Y_fast_list_values.append(k)

              for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
                  if k == np.max(self.Y_fast_list_values):
                      self.X_fastest_list_value.append(i)
                      self.Y_fastest_list_value.append(k)
                      #print(XFastest)
                      #print(YFastest)

              for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
                  if k == np.min(self.Y_slow_list_values):
                      self.X_slowest_list_value.append(i)
                      self.Y_slowest_list_value.append(k)

          result = func(self, *args, **kwargs)
    
          return result
      return wrapper
  
  @timer_func
  def barchart_horizontal(self) -> None:
    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""

    # *
    Horizontal = "horizontal"

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

    # * Get X and Y values
    X = list(self.Dataframe.iloc[:, 1])
    Y = list(self.Dataframe.iloc[:, self.Plot_column])

    plt.figure(figsize = (self.X_figure_size, self.Y_figure_size))

    # * Reverse is a bool variable with the postion of the plot
    if self.Plot_reverse == True:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self.X_fast_list_values.append(i)
                self.Y_fast_list_values.append(k)
            elif k >= np.mean(Y):
                self.X_slow_list_values.append(i)
                self.Y_slow_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
            if k == np.min(self.Y_fast_list_values):
                self.X_fastest_list_value.append(i)
                self.Y_fastest_list_value.append(k)
                #print(X_fastest_list_value)
                #print(Y_fastest_list_value)

        for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
            if k == np.max(self.Y_slow_list_values):
                self.X_slowest_list_value.append(i)
                self.Y_slowest_list_value.append(k)

    else:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self.X_slow_list_values.append(i)
                self.Y_slow_list_values.append(k)
            elif k >= np.mean(Y):
                self.X_fast_list_values.append(i)
                self.Y_fast_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
            if k == np.max(self.Y_fast_list_values):
                self.X_fastest_list_value.append(i)
                self.Y_fastest_list_value.append(k)
                #print(XFastest)
                #print(YFastest)

        for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
            if k == np.min(self.Y_slow_list_values):
                self.X_slowest_list_value.append(i)
                self.Y_slowest_list_value.append(k)

    # * Plot the data using bar() method
    plt.bar(self.X_slow_list_values, self.Y_slow_list_values, label = "Bad", color = 'gray')
    plt.bar(self.X_slowest_list_value, self.Y_slowest_list_value, label = "Worse", color = 'black')
    plt.bar(self.X_fast_list_values, self.Y_fast_list_values, label = "Better", color = 'lightcoral')
    plt.bar(self.X_fastest_list_value, self.Y_fastest_list_value, label = "Best", color = 'red')

    # *
    for Index, value in enumerate(self.Y_slowest_list_value):
        plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self.X_slowest_list_value[0])), fontweight = 'bold', fontsize = self.Font_size_general + 1)

    # *
    for Index, value in enumerate(self.Y_fastest_list_value):
        plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self.X_fastest_list_value[0])), fontweight = 'bold', fontsize = self.Font_size_general + 1)

    plt.legend(fontsize = self.Font_size_general)

    plt.title(self.Title, fontsize = self.Font_size_title)
    plt.xlabel(self.Plot_x_label, fontsize = self.Font_size_general)
    plt.xticks(fontsize = self.Font_size_ticks)
    plt.ylabel("Models", fontsize = self.Font_size_general)
    plt.yticks(fontsize = self.Font_size_ticks)
    plt.grid(color = self.Colors[0], linestyle = '-', linewidth = 0.2)

    # *
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()

    # *
    for i, value in enumerate(self.Y_slow_list_values):
        plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self.Font_size_ticks, color = 'black')

        Next_value = i

    Next_value = Next_value + 1

    for i, value in enumerate(self.Y_fast_list_values):
        plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self.Font_size_ticks, color = 'black')

    #plt.savefig(Graph_name_folder)

    self.save_figure(self.Save_figure, self.Title, Horizontal, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def barchart_vertical(self) -> None:  

    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""

    # *
    Vertical = "Vertical"

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

    # * Get X and Y values
    X = list(self.Dataframe.iloc[:, 1])
    Y = list(self.Dataframe.iloc[:, self.Plot_column])

    plt.figure(figsize = (self.X_figure_size, self.Y_figure_size))

    # * Reverse is a bool variable with the postion of the plot
    if self.Plot_reverse == True:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self.X_fast_list_values.append(i)
                self.Y_fast_list_values.append(k)
            elif k >= np.mean(Y):
                self.X_slow_list_values.append(i)
                self.Y_slow_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
            if k == np.min(self.Y_fast_list_values):
                self.X_fastest_list_value.append(i)
                self.Y_fastest_list_value.append(k)
                #print(X_fastest_list_value)
                #print(Y_fastest_list_value)

        for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
            if k == np.max(self.Y_slow_list_values):
                self.X_slowest_list_value.append(i)
                self.Y_slowest_list_value.append(k)

    else:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self.X_slow_list_values.append(i)
                self.Y_slow_list_values.append(k)
            elif k >= np.mean(Y):
                self.X_fast_list_values.append(i)
                self.Y_fast_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self.X_fast_list_values, self.Y_fast_list_values)):
            if k == np.max(self.Y_fast_list_values):
                self.X_fastest_list_value.append(i)
                self.Y_fastest_list_value.append(k)
                #print(XFastest)
                #print(YFastest)

        for Index, (i, k) in enumerate(zip(self.X_slow_list_values, self.Y_slow_list_values)):
            if k == np.min(self.Y_slow_list_values):
                self.X_slowest_list_value.append(i)
                self.Y_slowest_list_value.append(k)

    # * Plot the data using bar() method
    plt.bar(self.X_slow_list_values, self.Y_slow_list_values, label = "Bad", color = 'gray')
    plt.bar(self.X_slowest_list_value, self.Y_slowest_list_value, label = "Worse", color = 'black')
    plt.bar(self.X_fast_list_values, self.Y_fast_list_values, label = "Better", color = 'lightcoral')
    plt.bar(self.X_fastest_list_value, self.Y_fastest_list_value, label = "Best", color = 'red')

    # *
    for Index, value in enumerate(self.Y_slowest_list_value):
        plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self.X_slowest_list_value[0])), fontweight = 'bold', fontsize = self.Font_size_general + 1)

    # *
    for Index, value in enumerate(self.Y_fastest_list_value):
        plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self.X_fastest_list_value[0])), fontweight = 'bold', fontsize = self.Font_size_general + 1)

    plt.legend(fontsize = self.Font_size_general)

    plt.title(self.Title, fontsize = self.Font_size_title)
    plt.xlabel(self.Plot_x_label, fontsize = self.Font_size_general)
    plt.xticks(fontsize = self.Font_size_ticks)
    plt.ylabel("Models", fontsize = self.Font_size_general)
    plt.yticks(fontsize = self.Font_size_ticks)
    plt.grid(color = self.Colors[0], linestyle = '-', linewidth = 0.2)

    # *
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()

    # *
    for i, value in enumerate(self.Y_slow_list_values):
        plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self.Font_size_ticks, color = 'black')

        Next_value = i

    Next_value = Next_value + 1

    for i, value in enumerate(self.Y_fast_list_values):
        plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self.Font_size_ticks, color = 'black')

    #plt.savefig(Graph_name_folder)

    self.save_figure(self.Save_figure, self.Title, Vertical, self.Folder_path)
    self.show_figure(self.Show_image)

# ? Create class folders

class FigurePlot(FigureAdjust):
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

    # * 
    self.Labels = kwargs.get('labels', None)

    # * 
    self.CM_dataframe = kwargs.get('CMdf', None)
    self.History_dataframe = kwargs.get('Hdf', None)
    self.ROC_dataframe = kwargs.get('ROCdf', None)

    # *
    self.X_size_figure_subplot = 2
    self.Y_size_figure_subplot = 2

    # *
    self.Confusion_matrix_dataframe = pd.read_csv(self.CM_dataframe)
    self.History_data_dataframe = pd.read_csv(self.History_dataframe)
    
    self.Roc_curve_dataframes = []
    for Dataframe in self.ROC_dataframe:
      self.Roc_curve_dataframes.append(pd.read_csv(Dataframe))

    # *
    self.Accuracy = self.History_data_dataframe.accuracy.to_list()
    self.Loss = self.History_data_dataframe.loss.to_list()
    self.Val_accuracy = self.History_data_dataframe.val_accuracy.to_list()
    self.Val_loss = self.History_data_dataframe.val_loss.to_list()

    self.FPRs = []
    self.TPRs = []
    for i in range(len(self.Roc_curve_dataframes)):
      self.FPRs.append(self.Roc_curve_dataframes[i].FPR.to_list())
      self.TPRs.append(self.Roc_curve_dataframes[i].TPR.to_list())

  # * CSV_path attribute
  @property
  def CSV_path_property(self):
      return self.CSV_path

  @CSV_path_property.setter
  def CSV_path_property(self, New_value):
      self.CSV_path = New_value
  
  @CSV_path_property.deleter
  def CSV_path_property(self):
      print("Deleting CSV_path...")
      del self.CSV_path

  # * Roc_curve_dataframe attribute
  @property
  def Roc_curve_dataframe_property(self):
      return self.Roc_curve_dataframe

  @Roc_curve_dataframe_property.setter
  def Roc_curve_dataframe_property(self, New_value):
      self.Roc_curve_dataframe = New_value
  
  @Roc_curve_dataframe_property.deleter
  def Roc_curve_dataframe_property(self):
      print("Deleting Roc_curve_dataframe...")
      del self.Roc_curve_dataframe

  @timer_func
  def figure_plot_four(self) -> None: 

    # *
    Four_plot = 'Four_plot'

    # * Figure's size
    plt.figure(figsize = (self.X_figure_size, self.Y_figure_size))
    plt.suptitle(self.Title, fontsize = self.Font_size_title)
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 4)

    # * Confusion matrix heatmap
    sns.set(font_scale = self.Font_size_general)

    # *
    ax = sns.heatmap(self.Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self.Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # * Subplot training accuracy
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 1)
    plt.plot(self.Accuracy, label = 'Training Accuracy')
    plt.plot(self.Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    # * Subplot training loss
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 2)
    plt.plot(self.Loss, label = 'Training Loss')
    plt.plot(self.Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    # * FPR and TPR values for the ROC curve
    Auc = auc(self.FPRs[0], self.TPRs[0])

    # * Subplot ROC curve
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(self.FPRs[0], self.TPRs[0], label = 'Test' + '(area = {:.4f})'.format(Auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    
    self.save_figure(self.Save_figure, self.Title, Four_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def figure_plot_four_multiclass(self) -> None: 
    
    # * Colors for ROC curves
    Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

    # *
    Four_plot = 'Four_plot'
    Roc_auc = dict()


    # * Figure's size
    plt.figure(figsize = (self.X_figure_size, self.Y_figure_size))
    plt.suptitle(self.Title, fontsize = self.Font_size_title)
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 4)

    # * Confusion matrix heatmap
    sns.set(font_scale = self.Font_size_general)

    # *
    ax = sns.heatmap(self.Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self.Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # * Subplot training accuracy
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 1)
    plt.plot(self.Accuracy, label = 'Training Accuracy')
    plt.plot(self.Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    # * Subplot training loss
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 2)
    plt.plot(self.Loss, label = 'Training Loss')
    plt.plot(self.Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    # * FPR and TPR values for the ROC curve
    for i in range(len(self.Roc_curve_dataframes)):
      Roc_auc[i] = auc(self.FPRs[i], self.TPRs[i])

    # * Plot ROC curve
    plt.subplot(self.X_size_figure_subplot, self.Y_size_figure_subplot, 3)
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(self.Roc_curve_dataframes)):
      plt.plot(self.FPRs[i], self.TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self.Labels[i], Roc_auc[i]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    
    self.save_figure(self.Save_figure, self.Title, Four_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def figure_plot_CM(self) -> None:
    
    # *
    CM_plot = 'CM_plot'

    # *
    Confusion_matrix_dataframe = pd.read_csv(self.CM_dataframe)

    # * Figure's size
    plt.figure(figsize = (self.X_figure_size / 2, self.Y_figure_size / 2))
    plt.title('Confusion Matrix with {}'.format(self.Title))

    # * Confusion matrix heatmap
    sns.set(font_scale = self.Font_size_general)

    # *
    ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self.Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    self.save_figure(self.Save_figure, self.Title, CM_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def figure_plot_acc(self) -> None:

    # *
    ACC_plot = 'ACC_plot'

    # * Figure's size
    plt.figure(figsize = (self.X_figure_size / 2, self.Y_figure_size / 2))
    plt.title('Training and Validation Accuracy with {}'.format(self.Title))

    # * Plot training accuracy
    plt.plot(self.Accuracy, label = 'Training Accuracy')
    plt.plot(self.Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.xlabel('Epoch')

    self.save_figure(self.Save_figure, self.Title, ACC_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def figure_plot_loss(self) -> None:

    # *
    Loss_plot = 'Loss_plot'

    # * Figure's size
    
    plt.figure(figsize = (self.X_figure_size / 2, self.Y_figure_size / 2))
    plt.title('Training and Validation Loss with {}'.format(self.Title))

    # * Plot training loss
    plt.plot(self.Loss, label = 'Training Loss')
    plt.plot(self.Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.xlabel('Epoch')

    self.save_figure(self.Save_figure, self.Title, Loss_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  @timer_func
  def figure_plot_ROC_curve(self) -> None:
    
    # *
    ROC_plot = 'ROC_plot'

    # * Figure's size
    plt.figure(figsize = (self.X_figure_size / 2, self.Y_figure_size / 2))
    plt.title('ROC curve Loss with {}'.format(self.Title))

    # * FPR and TPR values for the ROC curve
    AUC = auc(self.FPRs[0], self.TPRs[0])

    # * Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(self.FPRs[0], self.TPRs[0], label = 'Test' + '(area = {:.4f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc = 'lower right')

    self.save_figure(self.Save_figure, self.Title, ROC_plot, self.Folder_path)
    self.show_figure(self.Show_image)

  def figure_plot_ROC_curve_multiclass(self) -> None:

    # * Colors for ROC curves
    Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

    # *
    ROC_plot = 'ROC_plot'
    Roc_auc = dict()

    # * Figure's size
    plt.figure(figsize = (self.X_figure_size / 2, self.Y_figure_size / 2))
    plt.title(self.Title, fontsize = self.Font_size_title)

    # * FPR and TPR values for the ROC curve
    for i in range(len(self.Roc_curve_dataframes)):
      Roc_auc[i] = auc(self.FPRs[i], self.TPRs[i])

    # * Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(self.Roc_curve_dataframes)):
      plt.plot(self.FPRs[i], self.TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self.Labels[i], Roc_auc[i]))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')

    self.save_figure(self.Save_figure, self.Title, ROC_plot, self.Folder_path)
    self.show_figure(self.Show_image)