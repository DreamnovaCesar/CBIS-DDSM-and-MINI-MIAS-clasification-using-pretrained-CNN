from Final_Code_0_0_Libraries import *
from Final_Code_1_General_Functions_Classes import Utilities

# ? define the standalone discriminator model

class ConfigurationGAN(Utilities):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Input_shape = kwargs.get('input_shape ', (224, 224, 3));

    @Utilities.timer_func
    @Utilities.detect_GPU
    def define_discriminator(self):
        """
        _summary_

        _extended_summary_

        Args:
            Input_shape (tuple, optional): _description_. Defaults to (224, 224, 3).

        Returns:
            _type_: _description_
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
