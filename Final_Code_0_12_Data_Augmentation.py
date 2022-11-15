from Final_Code_0_0_Libraries import *
from Final_Code_1_General_Functions_Classes import Utilities

# Data Augmentation function

class DataAugmentation(Utilities):
  
  @Utilities.timer_func
  def __init__(self, **kwargs) -> None:
    
    # * Instance attributes
    self.__Folder: str = kwargs.get('Folder', None)
    self.__Folder_dest: str = kwargs.get('NewFolder', None)
    self.__Severity: str = kwargs.get('Severity', None)
    self.__Sampling: int = kwargs.get('Sampling', 3)
    self.__Label: int = kwargs.get('Label', None)
    self.__Save_images: bool = kwargs.get('SI', False)

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

  def __repr__(self):

        kwargs_info = "Folder: {} , Folder_dest: {}, Severity: {}, Sampling: {}, Label: {}, Save_images: {}".format(  self.__Folder, self.__Folder_dest, self.__Severity,
                                                                                                                      self.__Sampling, self.__Label, self.__Save_images)
        return kwargs_info

  def __str__(self):

        Descripcion_class = ''
        
        return Descripcion_class
  
  # * Folder attribute
  @property
  def Folder_property(self):
      return self.__Folder

  @Folder_property.setter
  def Folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.__Folder = New_value
  
  @Folder_property.deleter
  def Folder_property(self):
      print("Deleting folder...")
      del self.__Folder

  # * Folder destination attribute
  @property
  def Folder_dest_property(self):
      return self.__Folder_dest

  @Folder_dest_property.setter
  def Folder_dest_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder dest must be a string") #! Alert
      self.__Folder_dest = New_value
  
  @Folder_dest_property.deleter
  def Folder_dest_property(self):
      print("Deleting destination folder...")
      del self.__Folder_dest

  # * Severity attribute
  @property
  def Severity_property(self):
      return self.__Severity

  @Severity_property.setter
  def Severity_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Severity must be a string") #! Alert
      self.__Severity = New_value
  
  @Severity_property.deleter
  def Severity_property(self):
      print("Deleting severity...")
      del self.__Severity

  # * Sampling attribute
  @property
  def Sampling_property(self):
      return self.__Sampling

  @Sampling_property.setter
  def Sampling_property(self, New_value):
    if not isinstance(New_value, int):
      raise TypeError("Must be a integer value ") #! Alert
    self.__Sampling = New_value
  
  @Sampling_property.deleter
  def Sampling_property(self):
      print("Deleting sampling...")
      del self.__Sampling

  # * Label attribute
  @property
  def Label_property(self):
      return self.__Label

  @Label_property.setter
  def Label_property(self, New_value):
    if (New_value > 10 or New_value < 0):
      raise ValueError("Value is out of the range must be less than 10 and more than 0") #! Alert
    if not isinstance(New_value, int):
      raise TypeError("Must be a enteger value") #! Alert
    self.__Label = New_value
  
  @Label_property.deleter
  def Label_property(self):
      print("Deleting label...")
      del self.__Label

  # * Save_images attribute
  @property
  def Save_images_property(self):
      return self.__Save_images

  @Save_images_property.setter
  def Save_images_property(self, New_value):
    if not isinstance(New_value, bool):
      raise TypeError("Must be a bool value (True or False is required)") #! Alert
    self.__Save_images = New_value
  
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

  # ? Flip horizontal using albumentation library

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

  # ? Flip vertical using albumentation library

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

  # ? Rotation using albumentation library

  @staticmethod
  def rotation(Rotation:int, Image_cropped: np.ndarray) -> np.ndarray:
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

  # ? shift rotation using albumentation library

  @Utilities.timer_func
  def data_augmentation(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data

    Args:
        self (_type_): _description_
        ndarray (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * Create a folder with each image and its transformations.

    #Name_dir:str = os.path.dirname(self.Folder)
    Name_base:str = os.path.basename(self.__Folder)

    #Name_dir_dest:str = os.path.dirname(self.Folder_dest)
    #Name_base_dest:str = os.path.basename(self.Folder_dest)
    #print(self.Folder_dest + '/' + Name_base + '_DA')

    Exist_dir:bool = os.path.isdir(self.__Folder_dest + '/' + Name_base + '_DA') 

    if self.__Save_images == True:
      if Exist_dir == False:
        New_folder_dest:str = self.__Folder_dest + '/' + Name_base + '_DA'
        os.mkdir(New_folder_dest)
      else:
        New_folder_dest:str = self.__Folder +  '/' + Name_base + '_DA'

    # * Lists to save the images and their respective labels
    Images:list = [] 
    Labels:list = [] 
    
    # * Initial value to rotate (More information on the albumentation's web)
    Rotation_initial_value:int = -120

    # * Reading the folder
    os.chdir(self.__Folder)
    Count:int = 1

    # * The number of images inside the folder
    Total_images:int = len(os.listdir(self.__Folder))

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}")
        Count += 1

        # * Resize with the given values
        Path_file:str = os.path.join(self.__Folder, File)
        Image:ndarray = cv2.imread(Path_file)

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Images.append(Image)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_Normal ✅'.format(Filename)
          #Filename_and_label = Filename + '_Normal'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(New_folder_dest, New_name_filename)

          io.imsave(New_folder, Image)
          
        # ? 1.A) Flip horizontal 

        Image_flip_horizontal = self.flip_horizontal(Image)

        Images.append(Image_flip_horizontal)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_FlipHorizontal_Augmentation ✅'.format(Filename)
          #Filename_and_label = Filename + '_FlipHorizontal' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(New_folder_dest, New_name_filename)

          io.imsave(New_folder, Image_flip_horizontal)

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_rotation = self.rotation(Rotation_initial_value, Image)

          Rotation_initial_value += 10

          Images.append(Image_rotation)
          Labels.append(self.__Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:
            
            Filename_and_label = '{}_{}_Rotation_Augmentation ✅'.format(Filename, str(i))
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(New_folder_dest, New_name_filename)

            io.imsave(New_folder, Image_rotation)

        # ? 2.A) Flip vertical

        Image_flip_vertical = self.flip_vertical(Image)

        Images.append(Image_flip_vertical)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:

          Filename_and_label = '{}_FlipVertical_Augmentation ✅'.format(Filename)
          #Filename_and_label = Filename + '_FlipVertical' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(New_folder_dest, New_name_filename)

          io.imsave(New_folder, Image_flip_vertical)
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_flip_vertical_rotation = self.rotation(Rotation_initial_value, Image_flip_vertical)

          Rotation_initial_value += 10

          Images.append(Image_flip_vertical_rotation)
          Labels.append(self.__Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:

            Filename_and_label = '{}_{}_Rotation_FlipVertical_Augmentation ✅'.format(Filename, str(i))
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(New_folder_dest, New_name_filename)

            io.imsave(New_folder, Image_flip_vertical_rotation)

    Labels = np.array(Labels)
    
    return Images, Labels
  
  @Utilities.timer_func
  def data_augmentation_same_folder(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data

    Args:
        self (_type_): _description_
        ndarray (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * Create a folder with each image and its transformations.

    #Name_dir:str = os.path.dirname(self.Folder)
    Name_base:str = os.path.basename(self.__Folder)

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
    Images:list = [] 
    Labels:list = [] 
    
    # * Initial value to rotate (More information on the albumentation's web)
    Rotation_initial_value:int = -120

    # * Reading the folder
    os.chdir(self.__Folder)
    Count:int = 1

    # * The number of images inside the folder
    Total_images:int = len(os.listdir(self.__Folder))

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}")
        Count += 1

        # * Resize with the given values
        Path_file:str = os.path.join(self.__Folder, File)
        Image:ndarray = cv2.imread(Path_file)

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Images.append(Image)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_Normal ✅'.format(Filename)
          #Filename_and_label = Filename + '_Normal'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.__Folder, New_name_filename)

          io.imsave(New_folder, Image)
          
        # ? 1.A) Flip horizontal 

        Image_flip_horizontal = self.flip_horizontal(Image)

        Images.append(Image_flip_horizontal)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:
          
          Filename_and_label = '{}_FlipHorizontal_Augmentation ✅'.format(Filename)
          #Filename_and_label = Filename + '_FlipHorizontal' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.__Folder, New_name_filename)

          io.imsave(New_folder, Image_flip_horizontal)

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_rotation = self.rotation(Rotation_initial_value, Image)

          Rotation_initial_value += 10

          Images.append(Image_rotation)
          Labels.append(self.__Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:
            
            Filename_and_label = '{}_{}_Rotation_Augmentation ✅'.format(Filename, str(i))
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(self.__Folder, New_name_filename)

            io.imsave(New_folder, Image_rotation)

        # ? 2.A) Flip vertical

        Image_flip_vertical = self.flip_vertical(Image)

        Images.append(Image_flip_vertical)
        Labels.append(self.__Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.__Save_images == True:

          Filename_and_label = '{}_FlipVertical_Augmentation ✅'.format(Filename)
          #Filename_and_label = Filename + '_FlipVertical' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.__Folder, New_name_filename)

          io.imsave(New_folder, Image_flip_vertical)
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.__Sampling):

          Image_flip_vertical_rotation = self.rotation(Rotation_initial_value, Image_flip_vertical)

          Rotation_initial_value += 10

          Images.append(Image_flip_vertical_rotation)
          Labels.append(self.__Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.__Save_images == True:

            Filename_and_label = '{}_{}_Rotation_FlipVertical_Augmentation ✅'.format(Filename, str(i))
            #Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(self.__Folder, New_name_filename)

            io.imsave(New_folder, Image_flip_vertical_rotation)

    Labels = np.array(Labels)
    
    return Images, Labels

  @Utilities.timer_func
  def data_augmentation_number_test_images(self) -> int:
    """
    Techniques used to increase the amount of data by adding slightly modified copies of already existing data 
    or newly created synthetic data from existing data. (Just the number of images).

    Args:
        ndarray (_type_): _description_

    Returns:
        _type_: _description_

    Applying data augmentation different transformations.

    Parameters:
    argument1 (folder): Folder chosen.
    argument2 (str): Severity of each image.
    argument3 (int): Amount of transformation applied for each image, using only rotation.
    argument4 (str): Label for each image.

    Returns:
    list:Returning images like 'X' value
    list:Returning labels like 'Y' value
    """
    
    Total_images_count = 0
  
    # * Reading the folder
    os.chdir(self.__Folder)
    Count = 1

    # * The number of images inside the folder
    Total_images = len(os.listdir(self.__Folder))

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.__Severity}")
        Count += 1

        # * Resize with the given values
        Path_file = os.path.join(self.__Folder, File)
        Image = cv2.imread(Path_file)

        #Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Total_images_count += 1

        # ? 1.A) Flip horizontal 

        Total_images_count += 1

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for _ in range(self.__Sampling):

          Total_images_count += 1

        # ? 2.A) Flip vertical

        Total_images_count += 1
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for _ in range(self.__Sampling):

          Total_images_count += 1

    return Total_images_count