
from Final_Code_0_0_Libraries import *
from Final_Code_0_1_Utilities import Utilities

################################################## ? Class decorators

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

# ? ####################################################### Mini-MIAS #######################################################

# ? Extract the mean of each column

@Utilities.timer_func
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

  List_data_mean = []

  for i in range(Dataframe.shape[0]):
      if Dataframe.iloc[i - 1, Column] > 0:
          List_data_mean.append(Dataframe.iloc[i - 1, Column])

  Mean_list:int = int(np.mean(List_data_mean))
  return Mean_list
 
# ? Clean Mini-MIAS CSV

@Utilities.timer_func
def mini_mias_csv_clean(Dataframe:pd.DataFrame) -> pd.DataFrame:
  """
  Clean the data from the Mini-MIAS dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe from Mini-MIAS' website.

  Returns:
      pd.DataFrame: Return the clean dataframe to use.
  """

  Value_fillna = 0
  # * This function will clean the data from the CSV archive

  Columns_list = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
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

# ? Kmeans algorithm
@Utilities.timer_func
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
  
@Utilities.timer_func
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

  #Dataframe_name = str(Technique_name) + '_Data_Removed_' + str(Severity) + '.csv'
  Dataframe_name = '{}_Data_Removed_{}.csv'.format(Technique_name, Severity)
  Dataframe_folder = os.path.join(Folder_CSV, Dataframe_name)

  Dataframe.to_csv(Dataframe_folder)

  return Dataframe


# ? ####################################################### CBIS-DDSM #######################################################

# ? CBIS-DDSM split data

@Utilities.timer_func
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

@Utilities.timer_func
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

    #remove_all_files(Folder_total_benign)
    #remove_all_files(Folder_benign)
    #remove_all_files(Folder_benign_wc)
    #remove_all_files(Folder_malignant)
    #remove_all_files(Folder_abnormal)

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

@Utilities.timer_func
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
@Utilities.timer_func
def concat_dataframe(*dfs: pd.DataFrame, **kwargs: str) -> pd.DataFrame:
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
  ALL_dataframes = [df for df in dfs]
  print(len(ALL_dataframes))
  Final_dataframe = pd.concat(ALL_dataframes, ignore_index = True, sort = False)
      
  #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
  #print(DataFrame)

  # * Name the final dataframe and save it into the given path

  if Save_file == True:
    #Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
    Dataframe_name = '{}_Dataframe_{}.csv'.format(str(Class_problem), str(Technique))
    Dataframe_folder_save = os.path.join(Folder_path, Dataframe_name)
    Final_dataframe.to_csv(Dataframe_folder_save)

  return Final_dataframe
