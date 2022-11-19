
from typing import Any
from Final_Code_0_0_Libraries import *

from Final_Code_0_0_Template_General_Functions_Classes import Utilities
from Final_Code_0_0_Template_General_Functions import FigureAdjust
from Final_Code_0_0_Template_General_Functions import FigurePlot

# ? Configuration of the CNN
class ConfigurationCNN(Utilities):
  """
    Utilities inheritance: 

    Methods:
        data_dic(): description

        @staticmethod
        overwrite_dic_CSV_folder(): description

        @staticmethod
        Model_pretrained(): description

        configuration_models_folder(): description
        
  """

  # * Initializing (Constructor)
  def __init__(self, **kwargs) -> None:
    """
    Keyword Args:
        folder (str): description 
        foldermodels (str): description
        foldermodelsesp (str): description
        foldercsv (str): description
        models (tuple(int)): description
        technique (str): description
        labels (list[str]): description
        X (int): description
        Y (int): description
        epochs (int): description
    """

    # * Instance attributes
    self.__Folder = kwargs.get('folder', None)
    self.__Folder_models = kwargs.get('foldermodels', None)
    self.__Folder_models_esp = kwargs.get('foldermodelsesp', None)
    self.__Folder_CSV = kwargs.get('foldercsv', None)

    self.__Models = kwargs.get('models', None)

    #Dataframe_save = kwargs.get('df', None)
    self.__Enhancement_technique = kwargs.get('technique', None)
    self.__Class_labels = kwargs.get('labels', None)
    #Column_names = kwargs.get('columns', None)

    self.__X_size = kwargs.get('X', None)
    self.__Y_size = kwargs.get('Y', None)
    self.__Epochs = kwargs.get('epochs', None)

    self.__Shape = (self.__X_size, self.__Y_size)
    self.__Batch_size = 32
    self.__Height_plot = 12
    self.__Width_plot  = 12
    
    self.__Color_mode  = 'rgb'

    # * Class problem definition
    self.__Classes = len(self.__Class_labels)

    # *
    if self.__Classes == 2:
      self.__Class_mode = "binary"
      self.__Class_problem_prefix = 'Biclass'
    elif self.__Classes >= 3:
      self.__Class_mode = "categorical"
      self.__Class_problem_prefix = 'Multiclass'
    else:
      raise TypeError("It can NOT be 1") #! Alert


  # * Class variables
  def __repr__(self):
        return f'[{self.__Folder}, {self.__Folder_models}, {self.__Folder_models_esp}, {self.__Folder_CSV}, {self.__Models}, {self.__Enhancement_technique}, {self.__Class_labels}, {self.__X_size}, {self.__Y_size}, {self.__Epochs}, {self.__Shape}, {self.__Batch_size}, {self.__Height_plot}, {self.__Width_plot}]';

  # * Class description
  def __str__(self):
      return  f'';
  
  # * Deleting (Calling destructor)
  def __del__(self):
      print('Destructor called, Configuration CNN class destroyed.');

  # * Get data from a dic
  def data_dic(self):

      return {'Folder path': str(self.__Folder),
              'Folder model': str(self.__Folder_models),
              'Folder model spanish': str(self.__Folder_models_esp),
              'Folder CSV': str(self.__Folder_CSV),
              'Models': str(self.__Models),
              'Enhancement technique': str(self.__Enhancement_technique),
              'Class labels': str(self.__Class_labels),
              'X size': str(self.__X_size),
              'Y size': str(self.__Y_size),
              'Epochs': str(self.__Epochs),
              'Shape': str(self.__Shape),
              'Batch size': str(self.__Batch_size),
              'Height (matplotlib)': str(self.__Height_plot),
              'Width (matplotlib)': str(self.__Width_plot),
              'Color mode': str(self.__Color_mode),
              };

  # ? Method to update CSV
  @staticmethod
  def overwrite_dic_CSV_folder(Dataframe: pd.DataFrame, Folder_path: str, Column_names: list[str], Column_values: list[str]):
      """
      Update the dataframe.

      Args:
          Dataframe (pd.DataFrame): Get the dataframe prepared to be update
          Folder_path (str): Folder path to save the dataframe
          Column_names (list[str]): Dataframe's headers
          Column_values (list[str]): Dataframe's values

      Returns:
          None
      """

      Row = dict(zip(Column_names, Column_values))
      Dataframe = Dataframe.append(Row, ignore_index = True)
    
      Dataframe.to_csv(Folder_path, index = False)
  
      print(Dataframe)

  # ? Method to choose the CNN model
  @staticmethod
  def Model_pretrained(X_size: int, Y_size: int, Num_classes: int, Model_pretrained_value: int):
    """
    Model configuration.

    Model_pretrained_value is a import variable that chose the model required.
    The next index show every model this function has:

    1:  EfficientNetB7_Model
    2:  EfficientNetB6_Model
    3:  EfficientNetB5_Model
    4:  EfficientNetB4_Model
    5:  EfficientNetB3_Model
    6:  EfficientNetB2_Model
    7:  EfficientNetB1_Model
    8:  EfficientNetB0_Model
    9:  ResNet50_Model
    10: ResNet50V2_Model
    11: ResNet152_Model
    12: ResNet152V2_Model
    13: MobileNet_Model
    14: MobileNetV3Small_Model
    15: MobileNetV3Large_Model
    16: Xception_Model
    17: VGG16_Model
    18: VGG19_Model
    19: InceptionV3_Model
    20: DenseNet121_Model
    21: DenseNet201_Model
    22: NASNetLarge_Model

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.
        Model_pretrained_value (int): Index of the model.

    Returns:
        string: Returning Model name.
        string: Returning Model name letters.
        Model: Returning the Model

    """
    # * Folder attribute (ValueError, TypeError)

    if not isinstance(Model_pretrained_value, int):
      raise TypeError("The index value must be integer") #! Alert

    # * Fine-Tuning MLP
    def MLP_classificador(x: Any, Units: int, Activation: str):
      """
      MLP configuration.

      Args:
          x (list): Layers.
          Units (int): The number of units for last layer.
          Activation (string): Activation used.

      Returns:
          _type_: _description_
      """
      x = Flatten()(x)
      x = BatchNormalization()(x)
      x = Dropout(0.5)(x)
      x = Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(0.01))(x)
      x = BatchNormalization()(x)
      x = Dropout(0.5)(x)
      #x = BatchNormalization()(x)
      x = Dense(Units, activation = Activation)(x)

      return x
    
    # * Function to choose pretrained model.
    def model_pretrained_index(Model_pretrained_value: int):

        if (Model_pretrained_value == 1):

            Model_name = 'EfficientNetB7_Model'
            Model_name_letters = 'ENB7'
            Model_index_chosen = EfficientNetB7
            
            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 2):

            Model_name = 'EfficientNetB6_Model'
            Model_name_letters = 'ENB6'
            Model_index_chosen = EfficientNetB6

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 3):

            Model_name = 'EfficientNetB5_Model'
            Model_name_letters = 'ENB5'
            Model_index_chosen = EfficientNetB5

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 4):

            Model_name = 'EfficientNetB4_Model'
            Model_name_letters = 'ENB4'
            Model_index_chosen = EfficientNetB4

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 5):
            
            Model_name = 'EfficientNetB3_Model'
            Model_name_letters = 'ENB3'
            Model_index_chosen = EfficientNetB3

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 6):
            
            Model_name = 'EfficientNetB2_Model'
            Model_name_letters = 'ENB2'
            Model_index_chosen = EfficientNetB2

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 7):
            
            Model_name = 'EfficientNetB1_Model'
            Model_name_letters = 'ENB1'
            Model_index_chosen = EfficientNetB1

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 8):
            
            Model_name = 'EfficientNetB0_Model'
            Model_name_letters = 'ENB0'
            Model_index_chosen = EfficientNetB0

            return Model_name, Model_name_letters, Model_index_chosen
        
        elif (Model_pretrained_value == 9):
            
            Model_name = 'ResNet50_Model'
            Model_name_letters = 'RN50'
            Model_index_chosen = ResNet50

            return Model_name, Model_name_letters, Model_index_chosen
        
        elif (Model_pretrained_value == 10):
            
            Model_name = 'ResNet50V2_Model'
            Model_name_letters = 'RN50V2'
            Model_index_chosen = ResNet50V2

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 11):
            
            Model_name = 'ResNet152_Model'
            Model_name_letters = 'RN152'
            Model_index_chosen = ResNet152

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 12):
            
            Model_name = 'ResNet152V2_Model'
            Model_name_letters = 'RN152V2'
            Model_index_chosen = ResNet152V2    

        elif (Model_pretrained_value == 13):
            
            Model_name = 'MobileNet_Model'
            Model_name_letters = 'MN'
            Model_index_chosen = MobileNet    

            return Model_name, Model_name_letters, Model_index_chosen
          
        elif (Model_pretrained_value == 14):
            
            Model_name = 'MobileNetV3Small_Model'
            Model_name_letters = 'MNV3S'
            Model_index_chosen = MobileNetV3Small    

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 15):
            
            Model_name = 'MobileNetV3Large_Model'
            Model_name_letters = 'MNV3L'
            Model_index_chosen = MobileNetV3Large   

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 16):
            
            Model_name = 'Xception_Model'
            Model_name_letters = 'Xc'
            Model_index_chosen = Xception

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 17):
            
            Model_name = 'VGG16_Model'
            Model_name_letters = 'VGG16'
            Model_index_chosen = VGG16

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 18):
            
            Model_name = 'VGG19_Model'
            Model_name_letters = 'VGG19'
            Model_index_chosen = VGG19

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 19):
            
            Model_name = 'InceptionV3_Model'
            Model_name_letters = 'IV3'
            Model_index_chosen = InceptionV3

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 20):
            
            Model_name = 'DenseNet121_Model'
            Model_name_letters = 'DN121'
            Model_index_chosen = DenseNet121

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 21):
            
            Model_name = 'DenseNet201_Model'
            Model_name_letters = 'DN201'
            Model_index_chosen = DenseNet201

            return Model_name, Model_name_letters, Model_index_chosen

        elif (Model_pretrained_value == 22):
            
            Model_name = 'NASNetLarge_Model'
            Model_name_letters = 'NNL'
            Model_index_chosen = NASNetLarge

            return Model_name, Model_name_letters, Model_index_chosen

        else:

          raise OSError("No model chosen") 

    Model_name, Model_name_letters, Model_index_chosen = model_pretrained_index(Model_pretrained_value)

    # *
    print('The model chosen is: {} -------- {} ✅'.format(Model_name, Model_name_letters))

    # *
    Model_input = Model_index_chosen(   input_shape = (X_size, Y_size, 3), 
                                        include_top = False, 
                                        weights = "imagenet")

    for layer in Model_input.layers:
        layer.trainable = False

    if Num_classes == 2:
        Activation = 'sigmoid'
        Loss = "binary_crossentropy"
        Units = 1
    else:
        Activation = 'softmax'
        Units = Num_classes
        Loss = "categorical_crossentropy"

    x = MLP_classificador(Model_input.output, Units, Activation)

    Model_CNN = Model(Model_input.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    Model_CNN.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ['accuracy']
    )

    return Model_name, Model_name_letters, Model_CNN

  # ? Method to change settings of the model
  @Utilities.timer_func
  @Utilities.detect_GPU
  def configuration_models_folder_CNN(self):
      """
      _summary_

      """

      # *
      Name_folder_training = '{}/train'.format(self.__Folder)
      Name_folder_val = '{}/val'.format(self.__Folder)
      Name_folder_test = '{}/test'.format(self.__Folder)

      # *
      train_datagen = ImageDataGenerator()
      val_datagen = ImageDataGenerator()
      test_datagen = ImageDataGenerator()


      # *
      Train_data = train_datagen.flow_from_directory(
          directory = Name_folder_training,
          target_size = self.__Shape,
          color_mode = self.__Color_mode,
          batch_size = self.__Batch_size,
          class_mode = self.__Class_mode,
          shuffle = True,
          seed = 42
      )

      # *
      Valid_data = val_datagen.flow_from_directory(
          directory = Name_folder_val,
          target_size = self.__Shape,
          color_mode = self.__Color_mode,
          batch_size = self.__Batch_size,
          class_mode = self.__Class_mode,
          shuffle = False,
          seed = 42        
      )

      # *
      Test_data = test_datagen.flow_from_directory(
          directory = Name_folder_test,
          target_size = self.__Shape,
          color_mode = self.__Color_mode,
          batch_size = self.__Batch_size,
          class_mode = self.__Class_mode,
          shuffle = False,
          seed = 42
      )

      for Index, Model in enumerate(self.__Models):
        
        # * Metrics digits
        Digits = 4

        # * List
        Info = []
        Dataframe_ROCs = []
        ROC_curve_FPR = []
        ROC_curve_TPR = []
        Classification_report_names = []

        # *
        Info_dic = {}

        # * Parameters dic classification report
        Macro_avg_label = 'macro avg'
        Weighted_avg_label = 'weighted avg'

        Classification_report_values = []
        Classification_report_labels = []
        Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

        for Label in self.__Class_labels:
          Classification_report_labels.append(Label)
        
        Classification_report_labels.append(Macro_avg_label)
        Classification_report_labels.append(Weighted_avg_label)

        #Precision_label = 'precision'
        #Recall_label = 'recall'
        #F1_score_label = 'f1-score'
        #Images_support_label = 'support'

        # * List
        Pretrained_model_name, Pretrained_model_name_letters, Pretrained_model = self.Model_pretrained(self.__X_size, self.__Y_size, self.__Classes, Model)

        #Pretrained_model_name_technique = str(Pretrained_model_name_letters) + '_' + str(Enhancement_technique)
        Pretrained_model_name_technique = "{}_{}".format(Pretrained_model_name_letters, self.__Enhancement_technique)

        # *
        Dir_name_csv = "{}_Folder_Data_Models_{}".format(self.__Class_problem_prefix, self.__Enhancement_technique)
        Dir_name_images = "{}_Folder_Images_Models_{}".format(self.__Class_problem_prefix, self.__Enhancement_technique)

        # *
        Dir_name_csv_model = "{}_Folder_Data_Model_{}_{}".format(self.__Class_problem_prefix, Pretrained_model_name_letters, self.__Enhancement_technique)
        Dir_name_images_model = "{}_Folder_Images_Model_{}_{}".format(self.__Class_problem_prefix, Pretrained_model_name_letters, self.__Enhancement_technique)

        # *
        Dir_data_csv = '{}/{}'.format(self.__Folder_CSV, Dir_name_csv)
        Dir_data_images = '{}/{}'.format(self.__Folder_CSV, Dir_name_images)

        # *
        Exist_dir_csv = os.path.isdir(Dir_data_csv)
        Exist_dir_images = os.path.isdir(Dir_data_images)

        # *
        if Exist_dir_csv == False:
          Folder_path = os.path.join(self.__Folder_CSV, Dir_name_csv)
          os.mkdir(Folder_path)
          print(Folder_path)
        else:
          Folder_path = os.path.join(self.__Folder_CSV, Dir_name_csv)
          print(Folder_path)

        if Exist_dir_images == False:
          Folder_path_images = os.path.join(self.__Folder_CSV, Dir_name_images)
          os.mkdir(Folder_path_images)
          print(Folder_path_images)
        else:
          Folder_path_images = os.path.join(self.__Folder_CSV, Dir_name_images)
          print(Folder_path_images)

        Dir_data_csv_model = '{}/{}'.format(Folder_path, Dir_name_csv_model)
        Dir_data_images_model = '{}/{}'.format(Folder_path_images, Dir_name_images_model)
        
        Exist_dir_csv_model = os.path.isdir(Dir_data_csv_model)
        Exist_dir_images_model = os.path.isdir(Dir_data_images_model)

        # *
        if Exist_dir_csv_model == False:
          Folder_path_in = os.path.join(Folder_path, Dir_name_csv_model)
          os.mkdir(Folder_path_in)
          print(Folder_path_in)
        else:
          Folder_path_in = os.path.join(Folder_path, Dir_name_csv_model)
          print(Folder_path_in)

        if Exist_dir_images_model == False:
          Folder_path_images_in = os.path.join(Folder_path_images, Dir_name_images_model)
          os.mkdir(Folder_path_images_in)
          print(Folder_path_images_in)
        else:
          Folder_path_images_in = os.path.join(Folder_path_images, Dir_name_images_model)
          print(Folder_path_images_in)


        # * Saving all data models the user gave.
        Dataframe_save_name = "{}_Dataframe_CNN_Folder_Data_{}_{}.csv".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        Dataframe_save_folder = os.path.join(Folder_path_in, Dataframe_save_name)

        # * Save best model weights for each model.
        Best_model_name_weights = "{}_Best_Model_Weights_{}_{}.h5".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        Best_model_folder_name_weights = os.path.join(Folder_path_in, Best_model_name_weights)
        
        # * Save dataframe CM for each model.
        #Confusion_matrix_dataframe_name = 'Dataframe_' + str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.csv'
        Confusion_matrix_dataframe_name = "{}_Dataframe_Confusion_Matrix_{}_{}.csv".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        Confusion_matrix_dataframe_folder = os.path.join(Folder_path_in, Confusion_matrix_dataframe_name)
        
        # * Save dataframe Logger (data: Accuracy and loss) for each model.
        #CSV_logger_info = str(Class_problem_prefix) + str(Pretrained_model_name) + '_' + str(Enhancement_technique) + '.csv'
        CSV_logger_info = "{}_Dataframe_Logger_{}_{}.csv".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        CSV_logger_info_folder = os.path.join(Folder_path_in, CSV_logger_info)

        # * Save dataframe ROC values for each model.
        Dataframe_ROC_name = "{}_Dataframe_ROC_Curve_Values_{}_{}.csv".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        Dataframe_ROC_folder = os.path.join(Folder_path_in, Dataframe_ROC_name)

        # * Save metrics images for each model
        #Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
        Image_name = "{}_{}_{}.png".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique)
        Image_folder = os.path.join(Folder_path_images_in, Image_name)

        # * Using ReduceLROnPlateau class.
        Model_reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                                            patience = 2, min_lr = 0.00001)

        # * Using ModelCheckpoint class.
        Model_checkpoint_callback = ModelCheckpoint(filepath = Best_model_folder_name_weights,
                                                    save_weights_only = True,                     
                                                    monitor = 'val_loss',
                                                    mode = 'max',
                                                    save_best_only = True )

        # * Using EarlyStopping class. 
        EarlyStopping_callback = EarlyStopping(patience = 2, monitor = 'val_loss')

        # * Using CSVLogger class to extract each epoch. 
        Log_CSV = CSVLogger(CSV_logger_info_folder, separator = ',', append = False)

        # * Save all callbacks to use them together
        Callbacks = [Model_checkpoint_callback, EarlyStopping_callback, Log_CSV]

        #################### * Training fit

        # * Use the fit to train the model (We used a timer to obtain the total time of the training)
        Start_training_time = time.time()

        Pretrained_Model_History = Pretrained_model.fit(  Train_data,
                                                          validation_data = Valid_data,
                                                          steps_per_epoch = Train_data.n//Train_data.batch_size,
                                                          validation_steps = Valid_data.n//Valid_data.batch_size,
                                                          epochs = self.__Epochs,
                                                          callbacks = Callbacks)

        #steps_per_epoch = Train_data.n//Train_data.batch_size,

        End_training_time = time.time()

        
        #################### * Test evaluation

        # * Use the evaluate (We used a timer to obtain the total time of the test)
        Start_testing_time = time.time()

        Loss_Test, Accuracy_Test = Pretrained_model.evaluate(Test_data)

        End_testing_time = time.time()
        
        #################### * Test evaluation

        # * Total time of training and testing
        Total_training_time = End_training_time - Start_training_time 
        Total_testing_time = End_testing_time - Start_testing_time
        
        # * Colums names for the dataframe created
        Column_names_ = [ 'name model', "model used", "accuracy training FE", "accuracy training LE", 
                          "accuracy testing", "loss train", "loss test", "training images", "validation images", 
                          "test images", "time training", "time testing", "technique used", "TN", "FP", "FN", "TP", "epochs", 
                          "precision", "recall", "f1_Score"]

        if self.__Classes == 2:

          Labels_biclass_number = []

          for i in range(len(self.__Class_labels)):
            Labels_biclass_number.append(i)

          # * Get the data from the model chosen

          Predict = Pretrained_model.predict(Test_data)
          y_pred = Pretrained_model.predict(Test_data).ravel()
          
          # * Biclass labeling
          y_pred_class = np.where(y_pred < 0.5, 0, 1)
          
          # * Confusion Matrix
          print('Confusion Matrix')
          Confusion_matrix = confusion_matrix(Test_data.classes, y_pred_class)
          
          print(Confusion_matrix)
          print(classification_report(Test_data.classes, y_pred_class, target_names = self.__Class_labels))
          
          # *
          #Report = classification_report(Test_data.classes, y_pred, target_names = self.__Class_labels)
          Dict = classification_report(Test_data.classes, y_pred, target_names = self.__Class_labels, output_dict = True)

          for i, Report_labels in enumerate(Classification_report_labels):
            for i, Metric_labels in enumerate(Classification_report_metrics_labels):
              
              # *
              Classification_report_names.append('{} {}'.format(Metric_labels, Report_labels))
              Classification_report_values.append(Dict[Report_labels][Metric_labels])
              #print(Classification_report_names)
              #print(Dict[Report_labels][Metric_labels])
          print("\n")

          # *
          Column_names_.extend(self.__Class_labels)
          Column_names_.extend(Classification_report_names)

          Dataframe_save = pd.DataFrame(columns = Column_names_)
          
          # * 
          Dataframe_save.to_csv(Dataframe_save_folder)

          # * Precision
          Precision = precision_score(Test_data.classes, y_pred_class)
          print(f"Precision: {round(Precision, Digits)}")
          print("\n")

          # * Recall
          Recall = recall_score(Test_data.classes, y_pred_class)
          print(f"Recall: {round(Recall, Digits)}")
          print("\n")

          # * F1-score
          F1_score = f1_score(Test_data.classes, y_pred_class)
          print(f"F1: {round(F1_score, Digits)}")
          print("\n")
          
          # *
          Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))
          Confusion_matrix_dataframe.to_csv(Confusion_matrix_dataframe_folder, index = False)

          # * FPR and TPR values for the ROC curve
          FPR, TPR, _ = roc_curve(Test_data.classes, y_pred_class)
          Auc = auc(FPR, TPR)
          
          # *
          for i in range(len(FPR)):
            ROC_curve_FPR.append(FPR[i])

          for i in range(len(TPR)):
            ROC_curve_TPR.append(TPR[i])

          # *
          Accuracy = Pretrained_Model_History.history['accuracy']
          Validation_accuracy = Pretrained_Model_History.history['val_accuracy']
          Loss = Pretrained_Model_History.history['loss']
          Validation_loss = Pretrained_Model_History.history['val_loss']

          # * Dict_roc_curve
          Dict_roc_curve = {'FPR': ROC_curve_FPR, 'TPR': ROC_curve_TPR} 
          Dataframe_ROC = pd.DataFrame(Dict_roc_curve)
          Dataframe_ROC.to_csv(Dataframe_ROC_folder)

          # *
          Plot_model = FigurePlot(folder = Folder_path_images_in, title = Pretrained_model_name, 
                                    SI = False, SF = True, CMdf = Confusion_matrix_dataframe_folder, 
                                      Hdf = CSV_logger_info_folder, ROCdf = [Dataframe_ROC_folder], labels = self.__Class_labels)

          # *
          Plot_model.figure_plot_four()
          Plot_model.figure_plot_CM()
          Plot_model.figure_plot_acc()
          Plot_model.figure_plot_loss()
          Plot_model.figure_plot_ROC_curve()

        elif self.__Classes >= 3:
          
          # * Dicts
          FPR = dict()
          TPR = dict()
          Roc_auc = dict()

          Labels_multiclass_number = []

          for i in range(len(self.__Class_labels)):
            Labels_multiclass_number.append(i)

          # * Get the data from the model chosen
          Predict = Pretrained_model.predict(Test_data)
          y_pred = Predict.argmax(axis = 1)

          # * Multiclass labeling
          y_pred_roc = label_binarize(y_pred, classes = Labels_multiclass_number)
          y_test_roc = label_binarize(Test_data.classes, classes = Labels_multiclass_number)

          # *
          #Report = classification_report(Test_data.classes, y_pred, target_names = Class_labels)
          Dict = classification_report(Test_data.classes, y_pred, target_names = self.__Class_labels, output_dict = True)

          for i, Report_labels in enumerate(Classification_report_labels):
            for i, Metric_labels in enumerate(Classification_report_metrics_labels):
              
              # *
              Classification_report_names.append('{} {}'.format(Metric_labels, Report_labels))
              Classification_report_values.append(Dict[Report_labels][Metric_labels])
              #print(Classification_report_names)
              #print(Dict[Report_labels][Metric_labels])
          print("\n")

          # *
          Column_names_.extend(self.__Class_labels)
          Column_names_.extend(Classification_report_names)
          
          # *
          Dataframe_save = pd.DataFrame(columns = Column_names_)

          # * 
          Dataframe_save.to_csv(Dataframe_save_folder)

          # * Confusion Matrix
          print('Confusion Matrix')
          Confusion_matrix = confusion_matrix(Test_data.classes, y_pred)

          print(Confusion_matrix)
          print(classification_report(Test_data.classes, y_pred, target_names = self.__Class_labels))

          # * Precision
          Precision = precision_score(Test_data.classes, y_pred, average = 'weighted')
          print(f"Precision: {round(Precision, Digits)}")
          print("\n")

          # * Recall
          Recall = recall_score(Test_data.classes, y_pred, average = 'weighted')
          print(f"Recall: {round(Recall, Digits)}")
          print("\n")

          # * F1-score
          F1_score = f1_score(Test_data.classes, y_pred, average = 'weighted')
          print(f"F1: {round(F1_score, Digits)}")
          print("\n")

          # *
          Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))
          Confusion_matrix_dataframe.to_csv(Confusion_matrix_dataframe_folder, index = False)

          # *
          for i in range(self.__Classes):
            FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
            Roc_auc[i] = auc(FPR[i], TPR[i])

          # *
          for i, FPR_row in enumerate(list(FPR.values())):
            ROC_curve_FPR.append(FPR_row)
            print(ROC_curve_FPR)
          for i, TPR_row in enumerate(list(TPR.values())):
            ROC_curve_TPR.append(TPR_row)
            print(ROC_curve_TPR)

          # *
          for j in range(len(ROC_curve_TPR)):

            Dict_roc_curve = {'FPR': ROC_curve_FPR[j], 'TPR': ROC_curve_TPR[j]}
            Dataframe_ROC = pd.DataFrame(Dict_roc_curve)

            Dataframe_ROC_name = "{}_Dataframe_ROC_Curve_Values_{}_{}_{}.csv".format(self.__Class_problem_prefix, Pretrained_model_name, self.__Enhancement_technique, j)
            Dataframe_ROC_folder = os.path.join(Folder_path_in, Dataframe_ROC_name)
            Dataframe_ROC.to_csv(Dataframe_ROC_folder)
            Dataframe_ROCs.append(Dataframe_ROC_folder)

          # *
          Accuracy = Pretrained_Model_History.history['accuracy']
          Validation_accuracy = Pretrained_Model_History.history['val_accuracy']
          Loss = Pretrained_Model_History.history['loss']
          Validation_loss = Pretrained_Model_History.history['val_loss']

          # *
          Plot_model = FigurePlot(folder = Folder_path_images_in, title = Pretrained_model_name, 
                                      SI = False, SF = True, CMdf = Confusion_matrix_dataframe_folder, 
                                        Hdf = CSV_logger_info_folder, ROCdf = [i for i in Dataframe_ROCs], labels = self.__Class_labels)

          # *
          Plot_model.figure_plot_four_multiclass()
          Plot_model.figure_plot_ROC_curve_multiclass()
          Plot_model.figure_plot_CM()
          Plot_model.figure_plot_acc()
          Plot_model.figure_plot_loss()

        Info.append(Pretrained_model_name_technique)
        Info.append(Pretrained_model_name)
        Info.append(Accuracy[self.__Epochs - 1])
        Info.append(Accuracy[0])
        Info.append(Accuracy_Test)
        Info.append(Loss[self.__Epochs - 1])
        Info.append(Loss_Test)

        Info.append(len(Train_data.classes))
        Info.append(len(Valid_data.classes))
        Info.append(len(Test_data.classes))

        Info.append(Total_training_time)
        Info.append(Total_testing_time)
        Info.append(self.__Enhancement_technique)

        Info.append(Confusion_matrix[0][0])
        Info.append(Confusion_matrix[0][1])
        Info.append(Confusion_matrix[1][0])
        Info.append(Confusion_matrix[1][1])

        Info.append(self.__Epochs)
        Info.append(Precision)
        Info.append(Recall)
        Info.append(F1_score)

        if self.__Classes == 2:
          Info.append(Auc)
        elif self.__Classes > 2:
          for i in range(self.__Classes):
            Info.append(Roc_auc[i])

        for i, value in enumerate(Classification_report_values):
          Info.append(value)

        print(Dataframe_save)

        #Dataframe_save = pd.read_csv(Dataframe_save_folder)
        self.overwrite_dic_CSV_folder(Dataframe_save, Dataframe_save_folder, Column_names_, Info)
      

# ? Function (Custom AlexNet12)
def CustomCNNAlexNet12_Model(X_size: int, Y_size: int, Num_classes: int):
    """
    CustomCNNAlexNet12 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning CustomCNNAlexNet12 model.
        string: Returning CustomCNNAlexNet12 Name.
    """

    Model_name = 'CustomAlexNet12_Model'
    Model_name_letters = 'CAN12'

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (X_size, Y_size, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(Units, activation = Activation)(x)

    CustomLeNet5_model = Model(CustomCNN_Model, x)
    
    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    CustomLeNet5_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5_model, Model_name, Model_name_letters

# ? Function (Custom AlexNet12 Tunner)
def CustomCNNAlexNet12Tunner_Model(X_size: int, Y_size: int, Num_classes: int, hp):
    """
    CustomCNNAlexNet12 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning CustomCNNAlexNet12 model.
        string: Returning CustomCNNAlexNet12 Name.
    """

    Model_name = 'CustomAlexNet12_Model'
    Model_name_letters = 'CAN12'

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (X_size, Y_size, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(Units, activation = Activation)(x)

    CustomLeNet5_model = Model(CustomCNN_Model, x)
    
    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    CustomLeNet5_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5_model, Model_name, Model_name_letters