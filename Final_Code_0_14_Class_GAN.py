from Final_Code_0_0_Libraries import *
from Final_Code_0_0_Template_General_Functions_Classes import Utilities

# ? define the standalone discriminator model

class ConfigurationGAN(Utilities):
    """
    Utilities inheritance: A class used to create image using a GAN.

    Methods:
        data_dic(): description

        define_discriminator(): description
        
    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Input_shape = kwargs.get('input_shape ', (224, 224, 3));

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder_path}, {self.__Input_shape}]';

    # * Class description
    def __str__(self):
        return  f'A class used to create image using a GAN.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, GAN class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder_path),
                'Input shape': str(self.__Input_shape),
                };

    # ? Method to define the GAN's discriminator
    @Utilities.timer_func
    @Utilities.detect_GPU
    def define_discriminator(self):
        """
        _summary_
        
        """
        model = Sequential()
        model.add(Conv2D(256, (3, 3), strides = (2, 2), padding = 'same', input_shape = self.__Input_shape))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        # compile model

        opt = Adam(lr = 0.0002, beta_1 = 0.5)
        model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
        return model
