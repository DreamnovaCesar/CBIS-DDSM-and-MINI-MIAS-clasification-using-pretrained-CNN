
from Final_Code_0_0_Libraries import *

from Final_Code_0_1_Class_Utilities import Utilities

class DCM_format(Utilities):
    """
    Utilities inheritance

    A class used to change the format DCM to png.

    Methods:
        data_dic(): description

        DCM_change_format(): description

    """
    # * Initializing (Constructor)
    def __init__(self, **kwargs: str) -> None:
        """
        Keyword Args:
            folder (str): description 
            AF (str): description
            PF (str): description
            PRF (str): description
            PRNF (str): description
            Severity (str): description
            Phase (str): description

        """
        # * This algorithm outputs crop values for images based on the coordinates of the CSV file.

        # * Instance attributes folders
        self.__Folder = kwargs.get('folder', None)
        self.__Folder_all = kwargs.get('AF', None)
        self.__Folder_patches = kwargs.get('PF', None)
        self.__Folder_resize = kwargs.get('PRF', None)
        self.__Folder_resize_normalize = kwargs.get('PRNF', None)

        # * Instance attributes labels
        self.__Severity = kwargs.get('Severity', None)
        self.__Phase = kwargs.get('Phase', None)

        # * Folder attribute (ValueError, TypeError)
        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder must be a string") #! Alert

        # * Folder destination where all the new images will be stored (ValueError, TypeError)
        if self.__Folder_all == None:
            raise ValueError("Destination folder does not exist") #! Alert
        if not isinstance(self.__Folder_all, str):
            raise TypeError("Destination folder must be a string") #! Alert

        # * Folder normal to stored images without preprocessing from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_patches == None:
            raise ValueError("Normal folder does not exist") #! Alert
        if not isinstance(self.__Folder_patches, str):
            raise TypeError("Normal folder must be a string") #! Alert

        # * Folder normal to stored resize images from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_resize == None:
            raise ValueError("Resize folder does not exist") #! Alert
        if not isinstance(self.__Folder_resize, str):
            raise TypeError("Resize folder must be a string") #! Alert

        # * Folder normal to stored resize normalize images from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_resize_normalize == None:
            raise ValueError("Normalize resize folder images does not exist") #! Alert
        if not isinstance(self.__Folder_resize_normalize, str):
            raise TypeError("Normalize resize folder must be a string") #! Alert

        # * Severity label (ValueError, TypeError)
        if self.__Severity == None:
            raise ValueError("Severity does not exist") #! Alert
        if not isinstance(self.__Severity, str):
            raise TypeError("Severity must be a string") #! Alert

        # * Phase label (ValueError, TypeError)
        if self.__Stage == None:
            raise ValueError("Phase images does not exist") #! Alert
        if not isinstance(self.__Stage, str):
            raise TypeError("Phase must be a string") #! Alert

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}, {self.__Folder_all}, {self.__Folder_patches}, {self.__Folder_resize}, {self.__Folder_resize_normalize}, {self.__Severity}, {self.__Phase}]';

    # * Class description
    def __str__(self):
        return  f'A class used to change the format DCM to png.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, DCM format class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder),
                'Folder all path': str(self.__Folder_all),
                'Folder patches': str(self.__Folder_patches),
                'Folder resize': str(self.__Folder_resize),
                'Folder normalise': str(self.__Folder_resize_normalize),
                'Severity': str(self.__Severity),
                'Phase': str(self.__Phase),
                };

    # * __Folder attribute
    @property
    def __Folder_property(self):
        return self.__Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        self.__Folder = New_value
    
    @__Folder_property.deleter
    def __Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    # * __Folder all images attribute
    @property
    def __Folder_all_property(self):
        return self.__Folder_all

    @__Folder_all_property.setter
    def __Folder_all_property(self, New_value):
        self.__Folder_all = New_value
    
    @__Folder_all_property.deleter
    def __Folder_all_property(self):
        print("Deleting all folder...")
        del self.__Folder_all

    # * __Folder patches images attribute
    @property
    def __Folder_patches_property(self):
        return self.__Folder_patches

    @__Folder_patches_property.setter
    def __Folder_patches_property(self, New_value):
        self.__Folder_patches = New_value
    
    @__Folder_patches_property.deleter
    def __Folder_patches_property(self):
        print("Deleting patches folder...")
        del self.__Folder_patches

    # * __Folder resize images attribute
    @property
    def __Folder_resize_property(self):
        return self.__Folder_resize

    @__Folder_resize_property.setter
    def __Folder_resize_property(self, New_value):
        self.__Folder_resize = New_value
    
    @__Folder_resize_property.deleter
    def __Folder_resize_property(self):
        print("Deleting resize folder...")
        del self.__Folder_resize

    # * __Folder resize normalize images attribute
    @property
    def __Folder_resize_normalize_property(self):
        return self.__Folder_resize_normalize

    @__Folder_resize_normalize_property.setter
    def __Folder_resize_normalize_property(self, New_value):
        self.__Folder_resize_normalize = New_value
    
    @__Folder_resize_normalize_property.deleter
    def Folder_resize_normalize_property(self):
        print("Deleting resize normalize folder...")
        del self.__Folder_resize_normalize

    # * __Severity attribute
    @property
    def __Severity_property(self):
        return self.__Severity

    @__Severity_property.setter
    def __Severity_property(self, New_value):
        self.__Severity = New_value
    
    @__Severity_property.deleter
    def __Severity_property(self):
        print("Deleting severity...")
        del self.__Severity
    
    # * __Stage
    @property
    def __Stage_property(self):
        return self.__Stage

    @__Stage_property.setter
    def __Stage_property(self, New_value):
        self.__Stage = New_value
    
    @__Stage_property.deleter
    def Stage_property(self):
        print("Deleting stage...")
        del self.__Stage

    # ? Method to change the DCM format
    @Utilities.timer_func
    def DCM_change_format(self) -> None:
        """
        Method to change DCM format to a PNG format. 

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
        Files_total = os.listdir(self.__Folder)

        # * Sorted files and multiply them
        Files_total_ = Files_total * 2
        Files_total_ = sorted(Files_total_)

        # * Search for each dir and file inside the folder given
        for Root, Dirs, Files in os.walk(self.__Folder, True):
            print("Root: {}".format(Root))
            print("Dirs: {}".format(Dirs))
            print("Files: {}".format(Files))
            print("-------------------------------")

        for Root, Dirs, Files in os.walk(self.__Folder):
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
        DCM_dataframe_name = 'DCM_Format_{}_{}.csv'.format(str(self.__Severity), str(self.__Stage))
        DCM_dataframe_folder = os.path.join(self.__Folder_all, DCM_dataframe_name)
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
            DCM_folder = os.path.join(self.__Folder_patches, DCM_name_file)
            DCM_folder_resize = os.path.join(self.__Folder_resize, DCM_name_file)
            DCM_folder_normalize = os.path.join(self.__Folder_resize_normalize, DCM_name_file)

            cv2.imwrite(DCM_folder, DCM_image_rescaled_float64)
            cv2.imwrite(DCM_folder_resize, DCM_image_resize)
            cv2.imwrite(DCM_folder_normalize, DCM_image_normalize)

            # * Print for comparison
            print('Images: ', DCM_Filenames[File], '------', Total_DCM_filenames[File])