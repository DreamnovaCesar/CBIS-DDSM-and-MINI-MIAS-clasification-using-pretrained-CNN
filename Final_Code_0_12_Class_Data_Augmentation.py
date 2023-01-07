from Final_Code_0_0_Libraries import *
from Final_Code_0_1_Class_Utilities import Utilities

# ? Data augmentation

class DataAugmentation(Utilities):
  """
    Utilities inheritance: A class used to increase the number of images synthetically using albumentation library

    Methods:
        data_dic(): description

        @staticmethod
        safe_rotation(Image_cropped: np.ndarray): Rotate the input by an angle selected randomly from the uniform distribution.

        @staticmethod
        flip_horizontal(Image_cropped: np.ndarray): Flip the input horizontally around the y-axis.

        @staticmethod
        flip_vertical(Image_cropped: np.ndarray): Flip the input vertically around the x-axis.

        @staticmethod
        rotation(Rotation: int, Image_cropped: np.ndarray): Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

        data_augmentation(): Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
        or newly created synthetic data from existing data (Saving the data in variables)

        data_augmentation_same_folder(): Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
        or newly created synthetic data from existing data (Saving the data in a folder)

        data_augmentation_number_test_images(): Just show the number of images created with a int
    """

  # * Initializing (Constructor)
  def __init__(self, **kwargs) -> None:
    """
    Keyword Args:
        folder (str): description 
        newfolder (str): description
        severity (str): description
        sampling (int): description
        label (int): description
        SI (bool): description
    """

    # * Instance attributes (kwargs)
    self.__Folder: str = kwargs.get('folder', None)
    self.__Folder_dest: str = kwargs.get('newfolder', None)
    self.__Severity: str = kwargs.get('severity', None)
    self.__Sampling: int = kwargs.get('sampling', 3)
    self.__Label: int = kwargs.get('label', None)
    self.__Save_images: bool = kwargs.get('SI', False)

    """
    # * Folder attribute (ValueError, TypeError)
    if (self.__Folder == None):
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.__Folder, str):
      raise TypeError("Folder must be a string") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if (self.__Folder_dest != None and self.__Save_images == True):
      if not isinstance(self.__Folder_dest, str):
        raise TypeError("Folder destination must be a string") #! Alert
    elif (self.__Folder_dest == None and self.__Save_images == True):
      warnings.warn('Saving the images is available but a folder destination was not found') #! Alert
      print("\n")
    elif (self.__Folder_dest != None and self.__Save_images == False):
      warnings.warn('Saving the images is unavailable but a folder destination was found') #! Alert
      print("\n")
    else:
      pass

    # * Severity attribute (ValueError, TypeError)
    if (self.__Severity == None):
      raise ValueError("Add the severity label") #! Alert
    if not isinstance(self.__Severity, str):
      raise TypeError("Severity label must be a string") #! Alert

    # * Sampling attribute (ValueError, TypeError)
    if not isinstance(self.__Sampling, int):
      raise TypeError("Sampling must be a integer") #! Alert

    # * Label attribute (ValueError, TypeError)
    if (self.__Label == None):
      raise ValueError("Must add the label") #! Alert
    if not isinstance(self.__Label, int):
      raise TypeError("Label must be a integer") #! Alert

    # * Save images attribute (ValueError, TypeError)
    if not isinstance(self.__Save_images, bool):
      raise TypeError("Save images attribute must be a bool value (True or False is required)") #! Alert
    """

  # * Class variables
  def __repr__(self):
        return f'[{self.__Folder}, {self.__Folder_dest}, {self.__Severity}, {self.__Sampling}, {self.__Label}, {self.__Save_images}]';

  # * Class description
  def __str__(self):
      return  f'A class used to increase the number of images synthetically using albumentation library';
  
  # * Deleting (Calling destructor)
  def __del__(self):
      print('Destructor called, data augmentation class destroyed.');

  # * Get data from a dic
  def data_dic(self):

      return {'Folder path': str(self.__Folder),
              'New folder path': str(self.__Folder_dest),
              'Severity': str(self.__Severity),
              'Sampling': str(self.__Sampling),
              'Labels': str(self.__Label),
              'Save images': str(self.__Save_images),
              };
  
  # * __Folder attribute
  @property
  def __Folder_property(self):
      return self.__Folder;

  @__Folder_property.setter
  def __Folder_property(self, New_value):
      self.__Folder = New_value;
  
  @__Folder_property.deleter
  def __Folder_property(self):
      print("Deleting folder...");
      del self.__Folder;

  # * __Folder_dest attribute
  @property
  def __Folder_dest_property(self):
      return self.__Folder_dest;

  @__Folder_dest_property.setter
  def __Folder_dest_property(self, New_value):
      self.__Folder_dest = New_value;
  
  @__Folder_dest_property.deleter
  def __Folder_dest_property(self):
      print("Deleting destination folder...");
      del self.__Folder_dest;

  # * __Severity attribute
  @property
  def __Severity_property(self):
      return self.__Severity;

  @__Severity_property.setter
  def __Severity_property(self, New_value):
      self.__Severity = New_value;
  
  @__Severity_property.deleter
  def __Severity_property(self):
      print("Deleting severity...");
      del self.__Severity;

  # * __Sampling attribute
  @property
  def __Sampling_property(self):
      return self.__Sampling;

  @__Sampling_property.setter
  def __Sampling_property(self, New_value):
    self.__Sampling = New_value;
  
  @__Sampling_property.deleter
  def __Sampling_property(self):
      print("Deleting sampling...");
      del self.__Sampling;

  # * __Label attribute
  @property
  def __Label_property(self):
      return self.__Label;

  @__Label_property.setter
  def __Label_property(self, New_value):
    if (New_value > 10 or New_value < 0):
      raise ValueError("Value is out of the range must be less than 10 and more than 0") #! Alert
    if not isinstance(New_value, int):
      raise TypeError("Must be a enteger value") #! Alert
    self.__Label = New_value;
  
  @__Label_property.deleter
  def __Label_property(self):
      print("Deleting label...");
      del self.__Label;

  # * __Save_images attribute
  @property
  def __Save_images_property(self):
      return self.__Save_images;

  @__Save_images_property.setter
  def __Save_images_property(self, New_value):
    if not isinstance(New_value, bool):
      raise TypeError("Must be a bool value (True or False is required)") #! Alert
    self.__Save_images = New_value;
  
  # ? Static method to safe rotation using albumentation library
  @staticmethod
  def safe_rotation(Image_cropped: np.ndarray) -> np.ndarray:
    """
    The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and after resizing, 
    it returns to its original shape with the original aspect ratio of the image. For these reason we may see some artifacts.
    Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        Image_cropped (ndarray): Raw image cropped that is use.

    Returns:
        ndarray: The image after the safe rotation transformation.
    """
    transform = A.Compose([
          A.ShiftScaleRotate(p = 1)
      ])
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Static method to flip horizontal using albumentation library
  @staticmethod
  def flip_horizontal(Image_cropped: np.ndarray) -> np.ndarray:
    """
    Flip the input horizontally around the y-axis.

    Args:
        Image_cropped (ndarray): Raw image cropped that is use.

    Returns:
        ndarray: The image after the flip horizontal transformation.
    """
    transform = A.Compose([
        A.HorizontalFlip(p = 1)
      ])
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Static method to flip vertical using albumentation library
  @staticmethod
  def flip_vertical(Image_cropped: np.ndarray) -> np.ndarray:
    """
    Flip the input vertically around the x-axis.

    Args:
        Image_cropped (ndarray): Raw image cropped that is use.

    Returns:
        ndarray: The image after the flip vertical transformation.
    """
    transform = A.Compose([
          A.VerticalFlip(p = 1)
        ])
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Static method to rotation using albumentation library
  @staticmethod
  def rotation(Rotation: int, Image_cropped: np.ndarray) -> np.ndarray:
    """
    Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

    Args:
        Rotation (int): Range from which a random angle is picked.
        Image_cropped (ndarray): Raw image cropped that is use.

    Returns:
        ndarray: The image after the rotation transformation.
    """
    transform = A.Compose([
        A.Rotate(Rotation, p = 1)
      ])
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Method to apply data augmentation using the sampling variable
  @Utilities.timer_func
  def data_augmentation(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data (Saving the data in variables)

    """

    # * Create a folder with each image and its transformations.

    #Name_dir:str = os.path.dirname(self.Folder)
    Name_base:str = os.path.basename(self.__Folder);

    #Name_dir_dest:str = os.path.dirname(self.Folder_dest)
    #Name_base_dest:str = os.path.basename(self.Folder_dest)
    #print(self.Folder_dest + '/' + Name_base + '_DA')

    Exist_dir:bool = os.path.isdir(self.__Folder_dest + '/' + Name_base + '_DA') ;

    if self.__Save_images == True:
      if Exist_dir == False:
        New_folder_dest:str = self.__Folder_dest + '/' + Name_base + '_DA';
        os.mkdir(New_folder_dest);
      else:
        New_folder_dest:str = self.__Folder +  '/' + Name_base + '_DA';

    # * Lists to save the images and their respective labels
    Images:list = [];
    Labels:list = [];
    
    # * Initial value to rotate (More information on the albumentation's web)
    Rotation_initial_value:int = -120

    # * Reading the folder
    os.chdir(self.__Folder);
    Count:int = 1;

    # * The number of images inside the folder
    Total_images:int = len(os.listdir(self.__Folder));

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File);

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}")
        Count += 1;

        # * Resize with the given values
        Path_file:str = os.path.join(self.__Folder, File)
        Image:ndarray = cv2.imread(Path_file)

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Images.append(Image);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_Normal ✅'.format(Filename);
          #Filename_and_label = Filename + '_Normal'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(New_folder_dest, New_name_filename);

          io.imsave(New_folder, Image);
          
        # ? 1.A) Flip horizontal 

        Image_flip_horizontal = self.flip_horizontal(Image)

        Images.append(Image_flip_horizontal);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_FlipHorizontal_Augmentation ✅'.format(Filename);
          #Filename_and_label = Filename + '_FlipHorizontal' + '_Augmentation'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(New_folder_dest, New_name_filename);

          io.imsave(New_folder, Image_flip_horizontal);

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_rotation = self.rotation(Rotation_initial_value, Image);

          Rotation_initial_value += 10;

          Images.append(Image_rotation);
          Labels.append(self.__Label);

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:
            
            Filename_and_label = '{}_{}_Rotation_Augmentation ✅'.format(Filename, str(i));
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            New_name_filename = Filename_and_label + Format;
            New_folder = os.path.join(New_folder_dest, New_name_filename);

            io.imsave(New_folder, Image_rotation);

        # ? 2.A) Flip vertical

        Image_flip_vertical = self.flip_vertical(Image);

        Images.append(Image_flip_vertical);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:

          Filename_and_label = '{}_FlipVertical_Augmentation ✅'.format(Filename);
          #Filename_and_label = Filename + '_FlipVertical' + '_Augmentation'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(New_folder_dest, New_name_filename);

          io.imsave(New_folder, Image_flip_vertical);
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_flip_vertical_rotation = self.rotation(Rotation_initial_value, Image_flip_vertical);

          Rotation_initial_value += 10;

          Images.append(Image_flip_vertical_rotation);
          Labels.append(self.__Label);

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:

            Filename_and_label = '{}_{}_Rotation_FlipVertical_Augmentation ✅'.format(Filename, str(i));
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            New_name_filename = Filename_and_label + Format;
            New_folder = os.path.join(New_folder_dest, New_name_filename);

            io.imsave(New_folder, Image_flip_vertical_rotation);

    Labels = np.array(Labels);
    
    return Images, Labels;
  
  # ? Method to apply data augmentation using the sampling variable saving them inside the folder
  @Utilities.timer_func
  def data_augmentation_same_folder(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data (Saving the data in a folder)

    """
    # * Create a folder with each image and its transformations.

    #Name_dir:str = os.path.dirname(self.Folder)
    #Name_base:str = os.path.basename(self.__Folder);

    #Name_dir_dest:str = os.path.dirname(self.Folder_dest)
    #Name_base_dest:str = os.path.basename(self.Folder_dest)
    #print(self.Folder_dest + '/' + Name_base + '_DA')

    """
    Exist_dir:bool = os.path.isdir(self.Folder_dest + '/' + Name_base + '_DA') 

    if self.Save_images == True:
      if Exist_dir == False:
        New_folder_dest:str = self.Folder_dest + '/' + Name_base + '_DA'
        os.mkdir(New_folder_dest)
      else:
        New_folder_dest:str = self.Folder +  '/' + Name_base + '_DA'

    """

    # * Lists to save the images and their respective labels
    Images:list = [];
    Labels:list = [];
    
    # * Initial value to rotate (More information on the albumentation's web)
    Rotation_initial_value:int = -120;

    # * Reading the folder
    os.chdir(self.__Folder);
    Count:int = 1;

    # * The number of images inside the folder
    Total_images:int = len(os.listdir(self.__Folder));

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File);

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}");
        Count += 1;

        # * Resize with the given values
        Path_file:str = os.path.join(self.__Folder, File);
        Image:ndarray = cv2.imread(Path_file);

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Images.append(Image);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_Normal ✅'.format(Filename);
          #Filename_and_label = Filename + '_Normal'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(self.__Folder, New_name_filename);

          io.imsave(New_folder, Image);
          
        # ? 1.A) Flip horizontal 

        Image_flip_horizontal = self.flip_horizontal(Image);

        Images.append(Image_flip_horizontal);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_FlipHorizontal_Augmentation ✅'.format(Filename);
          #Filename_and_label = Filename + '_FlipHorizontal' + '_Augmentation'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(self.__Folder, New_name_filename);

          io.imsave(New_folder, Image_flip_horizontal);

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_rotation = self.rotation(Rotation_initial_value, Image);

          Rotation_initial_value += 10;

          Images.append(Image_rotation);
          Labels.append(self.__Label);

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:
            
            Filename_and_label = '{}_{}_Rotation_Augmentation ✅'.format(Filename, str(i));
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            New_name_filename = Filename_and_label + Format;
            New_folder = os.path.join(self.__Folder, New_name_filename);

            io.imsave(New_folder, Image_rotation);

        # ? 2.A) Flip vertical

        Image_flip_vertical = self.flip_vertical(Image);

        Images.append(Image_flip_vertical);
        Labels.append(self.__Label);

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:

          Filename_and_label = '{}_FlipVertical_Augmentation ✅'.format(Filename);
          #Filename_and_label = Filename + '_FlipVertical' + '_Augmentation'
          New_name_filename = Filename_and_label + Format;
          New_folder = os.path.join(self.__Folder, New_name_filename);

          io.imsave(New_folder, Image_flip_vertical);
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_flip_vertical_rotation = self.rotation(Rotation_initial_value, Image_flip_vertical);

          Rotation_initial_value += 10;

          Images.append(Image_flip_vertical_rotation);
          Labels.append(self.__Label);

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:

            Filename_and_label = '{}_{}_Rotation_FlipVertical_Augmentation ✅'.format(Filename, str(i));
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            New_name_filename = Filename_and_label + Format;
            New_folder = os.path.join(self.__Folder, New_name_filename);

            io.imsave(New_folder, Image_flip_vertical_rotation);

    Labels = np.array(Labels)
    
    return Images, Labels

  # ? Method to apply data augmentation and get the number of images augmented
  @Utilities.timer_func
  def data_augmentation_number_test_images(self) -> int:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data (Saving the data in a folder).

    Just show the number of images created with a int
    
    """
    
    # * init count variable
    Total_images_count = 0;
  
    # * Reading the folder
    os.chdir(self.__Folder);
    Count = 1;

    # * The number of images inside the folder
    Total_images = len(os.listdir(self.__Folder));

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File);

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}");
        Count += 1;

        # * Resize with the given values
        Path_file = os.path.join(self.__Folder, File);
        Image = cv2.imread(Path_file);

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Total_images_count += 1;

        # ? 1.A) Flip horizontal 

        Total_images_count += 1;

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for _ in range(self.__Sampling):

          Total_images_count += 1;

        # ? 2.A) Flip vertical

        Total_images_count += 1;
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for _ in range(self.__Sampling):

          Total_images_count += 1;

    return Total_images_count;