from Final_Code_0_0_Libraries import *
from Final_Code_0_0_Template_General_Functions_Classes import Utilities

from Final_Code_0_0_Template_General_Functions import sort_images
from Final_Code_0_0_Template_General_Functions import remove_all_files

# ? Image processing

class ImageProcessing(Utilities):
  """
    Utilities inheritance: A class used to do the pre-processing of the images given such as: resize, median filter, Contrast Limited Adaptive Histogram Equalization (CLAHE), unsharp masking, gamma correction.

    Methods:
        data_dic(): description

        @staticmethod
        gamma_correction(): description

        resize_technique(): description

        normalize_technique(): description

        median_filter_technique(): description

        CLAHE_technique(): description

        histogram_equalization_technique(): description

        unsharp_masking_technique(): description

        contrast_stretching_technique(): description

        gamma_correction_technique(): description
        
  """

  # * Initializing (Constructor)
  def __init__(self, **kwargs) -> None:
    """
    Keyword Args:
        folder (str): description 
        newfolder (str): description
        severity (str): description
        label (int): description
        interpolation (int): description
        X (int): description
        Y (int): description
        division (int): description
        cliplimit (float): description
        radius (int): description
        amount (int): description
        GC (float): Gamma correction
    """

    # * General parameters
    self.__Folder = kwargs.get('folder', None);
    self.__New_folder = kwargs.get('newfolder', None);
    self.__Severity = kwargs.get('severity', None);
    self.__Label = kwargs.get('label', None);

    # * Parameters for resizing
    self.__Interpolation = kwargs.get('interpolation', cv2.INTER_CUBIC);
    self.__X_resize = kwargs.get('X', 224);
    self.__Y_resize = kwargs.get('Y', 224);

    # * Parameters for median filter
    self.__Division = kwargs.get('division', 3);

    # * Parameters for CLAHE
    self.__Clip_limit = kwargs.get('cliplimit', 0.01);

    # * Parameters for unsharp masking
    self.__Radius = kwargs.get('radius', 1);
    self.__Amount = kwargs.get('amount', 1);

    # * Parameters for gamma correction
    self.__Gamma_correction = kwargs.get('GC', 1.0);

    """
    # * Folder attribute (ValueError, TypeError)
    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.Folder, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * New folder attribute (ValueError, TypeError)
    if self.New_folder == None:
      raise ValueError("Folder destination attribute does not exist") #! Alert
    if not isinstance(self.New_folder, str):
      raise TypeError("Folder destination attribute must be a string") #! Alert

    # * Severity (ValueError, TypeError)
    if self.Severity == None:
      raise ValueError("Severity attribute does not exist") #! Alert
    if not isinstance(self.Severity, str):
      raise TypeError("Severity attribute must be a string") #! Alert

    # * Label (ValueError, TypeError)
    if self.Label == None:
      raise ValueError("Label attribute does not exist") #! Alert
    if not isinstance(self.Label, int):
      raise TypeError("Label attribute must be a string") #! Alert

    # ? Resize
    # * X resize (TypeError)
    if not isinstance(self.X_resize, int):
      raise TypeError("X must be integer") #! Alert
    # * Y resize (TypeError)
    if not isinstance(self.Y_resize, int):
      raise TypeError("Y must be integer") #! Alert

    # ? Median Filter
    # * Division (TypeError)
    if not isinstance(self.Division, int):
      raise TypeError("Division must be integer") #! Alert

    # ? CLAHE
    # * Clip limit (TypeError)
    if not isinstance(self.Clip_limit, float):
      raise TypeError("Clip limit must be float") #! Alert

    # ? Unsharp masking
    # * Radius and amount (TypeError)
    if not isinstance(self.Radius, int):
      raise TypeError("Clip limit must be integer") #! Alert
    if not isinstance(self.Amount, int):
      raise TypeError("Clip limit must be integer") #! Alert
    """

  # * Class variables
  def __repr__(self):
        return f'[{self.__Folder}, {self.__New_folder}, {self.__Severity}, {self.__Label}, {self.__Interpolation}, {self.__X_resize}, {self.__Y_resize}, {self.__Division}, {self.__Clip_limit}, {self.__Radius}, {self.__Amount}, {self.__Gamma_correction}]';

  # * Class description
  def __str__(self):
      return  f'A class used to do the pre-processing of the images given such as: resize, median filter, Contrast Limited Adaptive Histogram Equalization (CLAHE), unsharp masking, gamma correction.';
  
  # * Deleting (Calling destructor)
  def __del__(self):
      print('Destructor called, image processing class destroyed.');

  # * Get data from a dic
  def data_dic(self):

      return {'Folder path': str(self.__Folder),
              'New folder path': str(self.__New_folder),
              'Severity': str(self.__Severity),
              'Sampling': str(self.__Label),
              'Interpolation': str(self.__Interpolation),
              'X resize': str(self.__X_resize),
              'Y resize': str(self.__Y_resize),
              'Division': str(self.__Division),
              'Clip limit': str(self.__Clip_limit),
              'Radius': str(self.__Radius),
              'Amount': str(self.__Amount),
              'Gamma correction': str(self.__Gamma_correction),
              };

  # * __Folder attribute
  @property
  def __Folder_property(self):
      return self.__Folder

  @__Folder_property.setter
  def __Folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.__Folder = New_value
  
  @__Folder_property.deleter
  def __Folder_property(self):
      print("Deleting folder...")
      del self.__Folder

  # * __Folder destination attribute
  @property
  def __Folder_dest_property(self):
      return self.__Folder_dest

  @__Folder_dest_property.setter
  def __Folder_dest_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder dest must be a string") #! Alert
      self.__Folder_dest = New_value
  
  @__Folder_dest_property.deleter
  def __Folder_dest_property(self):
      print("Deleting destination folder...")
      del self.__Folder_dest

  # * Severity attribute
  @property
  def __Severity_property(self):
      return self.__Severity

  @__Severity_property.setter
  def __Severity_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Severity must be a string") #! Alert
      self.__Severity = New_value
  
  @__Severity_property.deleter
  def S__everity_property(self):
      print("Deleting severity...")
      del self.__Severity
  
  # * __Label attribute
  @property
  def __Label_property(self):
      return self.__Label

  @__Label_property.setter
  def __Label_property(self, New_value):
    if not isinstance(New_value, int):
      raise TypeError("Must be a integer value") #! Alert
    self.__Label = New_value
  
  @__Label_property.deleter
  def __Label_property(self):
      print("Deleting label...")
      del self.__Label

  # * __Interpolation attribute
  @property
  def __Interpolation_property(self):
      return self.__Interpolation

  @__Interpolation_property.setter
  def __Interpolation_property(self, New_value):
    self.__Interpolation = New_value
  
  @__Interpolation_property.deleter
  def __Interpolation_property(self):
      print("Deleting Interpolation...")
      del self.__Interpolation

  # * __X resize attribute
  @property
  def __X_resize_property(self):
      return self.__X_resize

  @__X_resize_property.setter
  def __X_resize_property(self, New_value):
    if not isinstance(New_value, int):
      raise TypeError("X must be a integer value") #! Alert
    self.__X_resize = New_value
  
  @__X_resize_property.deleter
  def __X_resize_property(self):
      print("Deleting X...")
      del self.__X_resize

  # * __Y resize attribute
  @property
  def __Y_resize_property(self):
      return self.Y_resize

  @__Y_resize_property.setter
  def __Y_resize_property(self, New_value):
    if not isinstance(New_value, int):
      raise TypeError("Y must be a integer value") #! Alert
    self.__Y_resize = New_value
  
  @__Y_resize_property.deleter
  def __Y_resize_property(self):
      print("Deleting Y...")
      del self.__Y_resize

  # * __Division attribute
  @property
  def __Division_property(self):
      return self.__Division

  @__Division_property.setter
  def __Division_property(self, New_value):
    if not isinstance(New_value, int):
      raise TypeError("Division must be a integer value") #! Alert
    self.__Division = New_value
  
  @__Division_property.deleter
  def __Division_property(self):
      print("Deleting division...")
      del self.__Division

  # * __Clip limit attribute
  @property
  def __Clip_limit_property(self):
      return self.__Clip_limit

  @__Clip_limit_property.setter
  def __Clip_limit_property(self, New_value):
    if not isinstance(New_value, float):
      raise TypeError("Clip limit must be a float value") #! Alert
    self.__Clip_limit = New_value
  
  @__Clip_limit_property.deleter
  def __Clip_limit_property(self):
      print("Deleting clip limit...")
      del self.__Clip_limit

  # * __Radius attribute
  @property
  def __Radius_property(self):
      return self.__Radius

  @__Radius_property.setter
  def __Radius_property(self, New_value):
    if not isinstance(New_value, float):
      raise TypeError("Radius must be a float value") #! Alert
    self.__Radius = New_value
  
  @__Radius_property.deleter
  def __Radius_property(self):
      print("Deleting Radius...")
      del self.__Radius

  # * Amount attribute
  @property
  def __Amount_property(self):
      return self.__Amount

  @__Amount_property.setter
  def __Amount_property(self, New_value):
    if not isinstance(New_value, float):
      raise TypeError("Amount must be a float value") #! Alert
    self.__Amount = New_value
  
  @__Amount_property.deleter
  def __Amount_property(self):
      print("Deleting Amount...")
      del self.__Amount

  # * __Gamma_correction attribute
  @property
  def __Gamma_correction_property(self):
      return self.__Gamma_correction

  @__Gamma_correction_property.setter
  def __Gamma_correction_property(self, New_value):
    if not isinstance(New_value, float):
      raise TypeError("Gamma_correction must be a float value") #! Alert
    self.__Gamma_correction = New_value
  
  @__Gamma_correction_property.deleter
  def __Gamma_correction_property(self):
      print("Deleting Gamma_correction...")
      del self.__Gamma_correction

  # ? Method to get gamma correction values
  @staticmethod
  def gamma_correction(Image, Gamma_value):
    """
    Summary

    Args:
        Image (ndarray): description
        Gamma_value (float): description

    Returns:
        None
    """

    Gamma_inv = 1 / Gamma_value

    Table_gamma = [((i / 255) ** Gamma_inv) * 255 for i in range(256)]
    Table_gamma = np.array(Table_gamma, np.uint8)

    return cv2.LUT(Image, Table_gamma)

  # ? Method to use resize technique
  @Utilities.timer_func
  def resize_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Save the new images in a list
    #New_images = [] 

    os.chdir(self.__Folder)
    print(os.getcwd())
    print("\n")

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count:int = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)
      
      if File.endswith(Format):

        try:
          
          print('Working with {} of {} images ✅'.format(Count, Total_images))
          Count += 1
          
          # * Resize with the given values 
          Path_file = os.path.join(self.__Folder, File)
          Imagen = cv2.imread(Path_file)

          # * Resize with the given values 
          Shape = (self.__X_resize, self.__Y_resize)
          Resized_imagen = cv2.resize(Imagen, Shape, interpolation = self.__Interpolation)

          # * Show old image and new image
          print(Imagen.shape, ' -------- ', Resized_imagen.shape)

          # * Name the new file
          New_name_filename = Filename + Format
          New_folder = os.path.join(self.__Folder, New_name_filename)

          # * Save the image in a new folder
          cv2.imwrite(New_folder, Resized_imagen)
          #New_images.append(Resized_Imagen)
          
        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ❌'.format(str(Count), str(Total_images))) #! Alert

  # ? Method to use normalization technique
  @Utilities.timer_func
  def normalize_technique(self):
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)

    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []
    
    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count:int = 1

    # * Reading the files
    for File in Sorted_files:

      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format): 
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))
          #print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = cv2.imread(Path_file)

          Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

          #print("%s has %d and %d" % (File, Imagen.shape[0], Imagen.shape[1]))

          # * Add a black image for normalization
          Image_numpy_zeros = np.zeros((Image.shape[0], Image.shape[1]))
          Normalization_imagen = cv2.normalize(Image, Image_numpy_zeros, 0, 255, cv2.NORM_MINMAX)

          # * Save each statistic in a variable
          Mae = mae(Image, Normalization_imagen)
          Mse = mse(Image, Normalization_imagen)
          Ssim = ssim(Image, Normalization_imagen)
          Psnr = psnr(Image, Normalization_imagen)
          Nrmse = nrmse(Image, Normalization_imagen)
          Nmi = nmi(Image, Normalization_imagen)
          R2s = r2s(Image, Normalization_imagen)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_Normalization'.format(str(Filename))
          #Filename_and_technique = Filename + '_Normalization'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)
          
          #Normalization_Imagen = Normalization_Imagen.astype('float32')
          #Normalization_Imagen = Normalization_Imagen / 255.0

          # * Save the image in a new folder
          cv2.imwrite(New_folder, Normalization_imagen)
          
          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          All_filenames.append(Filename_and_technique)
          Labels.append(self.__Label)

          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe

  # ? Method to use median filter technique
  @Utilities.timer_func
  def median_filter_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)

    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    # *
    os.chdir(self.__Folder)

    # *
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          #Image_median_filter = cv2.medianBlur(Imagen, Division)
          Median_filter_image = filters.median(Image, np.ones((self.__Division, self.__Division)))

          # * Save each statistic in a variable
          Mae = mae(Image, Median_filter_image)
          Mse = mse(Image, Median_filter_image)
          Ssim = ssim(Image, Median_filter_image)
          Psnr = psnr(Image, Median_filter_image)
          Nrmse = nrmse(Image, Median_filter_image)
          Nmi = nmi(Image, Median_filter_image)
          R2s = r2s(Image, Median_filter_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_Median_Filter'.format(str(Filename))
          #Filename_and_technique = Filename + '_Median_Filter'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, Median_filter_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)

          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

  # ? Method to use CLAHE technique 
  @Utilities.timer_func
  def CLAHE_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
          #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
          #CLAHE_Imagen = CLAHE.apply(Imagen)

          CLAHE_image = equalize_adapthist(Image, clip_limit = self.__Clip_limit)

          Image = img_as_ubyte(Image)
          CLAHE_image = img_as_ubyte(CLAHE_image)

          # * Save each statistic in a variable
          Mae = mae(Image, CLAHE_image)
          Mse = mse(Image, CLAHE_image)
          Ssim = ssim(Image, CLAHE_image)
          Psnr = psnr(Image, CLAHE_image)
          Nrmse = nrmse(Image, CLAHE_image)
          Nmi = nmi(Image, CLAHE_image)
          R2s = r2s(Image, CLAHE_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_CLAHE'.format(str(Filename))
          #Filename_and_technique = Filename + '_CLAHE'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, CLAHE_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)

          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

  # ? Method to use the histogram equalization technique
  @Utilities.timer_func
  def histogram_equalization_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          HE_image = equalize_hist(Image)

          Image = img_as_ubyte(Image)
          HE_image = img_as_ubyte(HE_image)

          # * Save each statistic in a variable
          Mae = mae(Image, HE_image)
          Mse = mse(Image, HE_image)
          Ssim = ssim(Image, HE_image)
          Psnr = psnr(Image, HE_image)
          Nrmse = nrmse(Image, HE_image)
          Nmi = nmi(Image, HE_image)
          R2s = r2s(Image, HE_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_HE'.format(str(Filename))
          #Filename_and_technique = Filename + '_HE'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, HE_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

  # ? Method to use unsharp masking technique 
  @Utilities.timer_func
  def unsharp_masking_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          UM_image = unsharp_mask(Image, radius = self.__Radius, amount = self.__Amount)

          Image = img_as_ubyte(Image)
          UM_image = img_as_ubyte(UM_image)

          # * Save each statistic in a variable
          Mae = mae(Image, UM_image)
          Mse = mse(Image, UM_image)
          Ssim = ssim(Image, UM_image)
          Psnr = psnr(Image, UM_image)
          Nrmse = nrmse(Image, UM_image)
          Nmi = nmi(Image, UM_image)
          R2s = r2s(Image, UM_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_UM'.format(str(Filename))
          #Filename_and_technique = Filename + '_UM'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, UM_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe

  # ? Method to use the contrast stretching technique 
  @Utilities.timer_func
  def contrast_stretching_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          p2, p98 = np.percentile(Image, (2, 98))
          CS_image = rescale_intensity(Image, in_range = (p2, p98))

          Image = img_as_ubyte(Image)
          CS_image = img_as_ubyte(CS_image)

          # * Save each statistic in a variable
          Mae = mae(Image, CS_image)
          Mse = mse(Image, CS_image)
          Ssim = ssim(Image, CS_image)
          Psnr = psnr(Image, CS_image)
          Nrmse = nrmse(Image, CS_image)
          Nmi = nmi(Image, CS_image)
          R2s = r2s(Image, CS_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_CS'.format(str(Filename))
          #Filename_and_technique = Filename + '_CS'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, CS_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe
  
  # ? Method to use the Gamma correction technique
  @Utilities.timer_func
  def gamma_correction_technique(self) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_
    """

    # * Remove all the files in the new folder using this function
    remove_all_files(self.__New_folder)
    
    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.__Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.__Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):
        
        try:
          print('Working with {} ✅'.format(Filename))
          print('Working with {} of {} {} images ✅'.format(Count, Total_images, self.__Severity))

          # * Resize with the given values
          Path_file = os.path.join(self.__Folder, File)
          Image = cv2.imread(Path_file, as_gray = True)

          # * Gamma correction 1.0 standard
          Image_gamma = self.gamma_correction(Image, self.__Gamma_correction)

          Image = img_as_ubyte(Image)
          Image_gamma = img_as_ubyte(Image_gamma)

          # * Save each statistic in a variable
          Mae = mae(Image, Image_gamma)
          Mse = mse(Image, Image_gamma)
          Ssim = ssim(Image, Image_gamma)
          Psnr = psnr(Image, Image_gamma)
          Nrmse = nrmse(Image, Image_gamma)
          Nmi = nmi(Image, Image_gamma)
          R2s = r2s(Image, Image_gamma)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = '{}_GC'.format(str(Filename))
          #Filename_and_technique = Filename + '_CS'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.__New_folder, New_name_filename)

          # * Save the image in a new folder
          cv2.imwrite(New_folder, Image_gamma)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.__Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

        except OSError:
          print('Cannot convert {} ❌'.format(str(File))) #! Alert

    print("\n")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe