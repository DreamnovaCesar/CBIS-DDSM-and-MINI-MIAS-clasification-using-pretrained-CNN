from Final_Code_0_0_Libraries import *

from typing import Any

from Final_Code_0_0_Template_General_Functions_Classes import Utilities
from Final_Code_0_9_Class_Figure_Adjust import FigurePlot

# ? Configuration of the CNN
class ConfigurationML(Utilities):
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
            EXT (str): Extract features technique
            technique (str): description
            labels (list[str]): description
            X (int): description
            Y (int): description
            epochs (int): description
        """

        # * Instance attributes
        self.__Folder = kwargs.get('folder', None)
        #self.__Folder_models = kwargs.get('foldermodels', None)
        #self.__Folder_models_esp = kwargs.get('foldermodelsesp', None)
        #self.__Folder_CSV = kwargs.get('foldercsv', None)


        self.__Dataframe = kwargs.get('dataframe', None)
        #self.__Dataframe_save = kwargs.get('dataframesave', None)
        self.__Models = kwargs.get('models', None)

        #Dataframe_save = kwargs.get('df', None)
        self.__Extract_feature_technique = kwargs.get('EXT', None)
        self.__Enhancement_technique = kwargs.get('technique', None)

        self.__Class_labels = kwargs.get('labels', None)
        #Column_names = kwargs.get('columns', None)

        #self.__X_size = kwargs.get('X', None)
        #self.__Y_size = kwargs.get('Y', None)
        self.__Epochs = kwargs.get('epochs', None)

        #self.__Shape = (self.__X_size, self.__Y_size)
        self.__Batch_size = 32
        #self.__Height_plot = 12
        #self.__Width_plot  = 12

        self.__sm = SMOTE()
        self.__sc = StandardScaler()

        # * Class problem definition
        self.__Classes = len(self.__Class_labels)

        if self.__Classes == 2:
            self.__Class_problem_prefix = 'Biclass'
        elif self.__Classes >= 3:
            self.__Class_problem_prefix = 'Multiclass'
        else:
            raise TypeError("It can NOT be 1") #! Alert

        if isinstance(self.__Dataframe, str):
            self.__Dataframe = pd.read_csv(self.__Dataframe)

    # * Class variables
    def __repr__(self):
            return f'[{self.__Folder}, {self.__Models}, {self.__Extract_feature_technique}, {self.__Enhancement_technique}, {self.__Class_labels}, {self.__Batch_size}, {self.__Height_plot}, {self.__Width_plot}]';

    # * Class description
    def __str__(self):
        return  f'';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Configuration ML class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self.__Folder),
                'Models': str(self.__Models),
                'Extract features technique': str(self.__Extract_feature_technique),
                'Enhancement technique': str(self.__Enhancement_technique),
                'Class labels': str(self.__Class_labels),
                'Batch size': str(self.__Batch_size),
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
    def Model_pretrained_ML(Model_pretrained_value: int, X_train, y_train, X_test):
        """
        Model configuration.

        Model_pretrained_value is a import variable that chose the model required.
        The next index show every model this function has:

        1: Support Vector Machine
        2: Multi Support Vector Machine
        3: Multi Layer Perceptron
        4: Decision Tree
        5: K Neighbors
        6: Random Forest
        7: Gradient Boostin Classifier

        Args:
            X_size (int): X's size value.
            Y_size (int): Y's size value.
            Num_classes (int): Number total of classes.
            Model_pretrained_value (int): Index of the model.

        Returns:
            _type_: _description_
            string: Returning Model.
            string: Returning Model Name.
        """
        # * Folder attribute (ValueError, TypeError)

        if not isinstance(Model_pretrained_value, int):
            raise TypeError("The index value must be integer") #! Alert

        # * Function to choose pretrained model.
        def model_pretrained_index(Model_pretrained_value: int):

            if (Model_pretrained_value == 1):

                Model_name = 'Support Vector Machine'
                Model_name_letters = 'SVM'
                Model_index_chosen = SVC(kernel = 'rbf', C = 1)
                
                return Model_name, Model_name_letters, Model_index_chosen

            elif (Model_pretrained_value == 2):

                Model_name = 'Multi Support Vector Machine'
                Model_name_letters = 'MSVM'
                Model_index_chosen = OneVsRestClassifier(SVC(kernel = 'rbf', C = 1))

                return Model_name, Model_name_letters, Model_index_chosen

            elif (Model_pretrained_value == 3):

                Model_name = 'Multi Layer Perceptron'
                Model_name_letters = 'MLP'
                Model_index_chosen = MLPClassifier(hidden_layer_sizes = [100] * 2, random_state = 1, max_iter = 2000)
            
                return Model_name, Model_name_letters, Model_index_chosen

            elif (Model_pretrained_value == 4):

                Model_name = 'Decision Tree'
                Model_name_letters = 'DT'
                Model_index_chosen = DecisionTreeClassifier(max_depth = 50)

                return Model_name, Model_name_letters, Model_index_chosen
            
            elif (Model_pretrained_value == 5):

                Model_name = 'K Neighbors'
                Model_name_letters = 'KNN'
                Model_index_chosen = KNeighborsClassifier(n_neighbors = 7)

                return Model_name, Model_name_letters, Model_index_chosen

            elif (Model_pretrained_value == 6):

                Model_name = 'Random Forest'
                Model_name_letters = 'RF'
                Model_index_chosen = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)

                return Model_name, Model_name_letters, Model_index_chosen
            
            elif (Model_pretrained_value == 7):

                Model_name = 'Gradient Boostin Classifier'
                Model_name_letters = 'GBC'
                Model_index_chosen = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 2, random_state = 0)

                return Model_name, Model_name_letters, Model_index_chosen

            else:

                raise OSError("No model chosen") 

        Model_name, Model_name_letters, Model_index_chosen = model_pretrained_index(Model_pretrained_value)

        print('The model chosen is: {} -------- {} ✅'.format(Model_name, Model_name_letters))

        Start_training_time = time.time()
        
        classifier = Model_index_chosen
        classifier.fit(X_train, y_train)

        End_training_time = time.time()

        Total_time_training = End_training_time - Start_training_time

        Y_pred = classifier.predict(X_test)

        return Model_name, Model_name_letters, Total_time_training, Y_pred

    # ? Method to change settings of the model
    @Utilities.timer_func
    def configuration_models_folder_ML(self):

        for Index, Model in enumerate(self.__Models):
            
            # * Metrics digits
            Digits = 4

            # * List
            Info = []
            Dataframe_ROCs = []
            ROC_curve_FPR = []
            ROC_curve_TPR = []
            Classification_report_labels = []
            Classification_report_values = []
            Classification_report_names = []

            # * Parameters dic classification report
            Macro_avg_label = 'macro avg'
            Weighted_avg_label = 'weighted avg'

            Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

            for Label in self.__Class_labels:
                Classification_report_labels.append(Label)
            
            Classification_report_labels.append(Macro_avg_label)
            Classification_report_labels.append(Weighted_avg_label)
            
            # * Extract data and label
            Dataframe_len_columns = len(self.__Dataframe.columns)

            X_total = self.__Dataframe.iloc[:, 0:Dataframe_len_columns - 1].values
            Y_total = self.__Dataframe.iloc[:, -1].values

            print(X_total)
            print(Y_total)

            #pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
            #print(Dataframe)

            # * Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(np.array(X_total), np.array(Y_total), test_size = 0.2, random_state = 1)

            # * Resample data for training
            X_train, y_train = self.__sm.fit_resample(X_train, y_train)

            # * Scaling data for training
            X_train = self.__sc.fit_transform(X_train)
            X_test = self.__sc.transform(X_test)

            #print(y_train)
            #print(y_test)

            # * Get the data from the model chosen
            Model_name, Model_name_letters, Total_time_training, Y_pred = self.Model_pretrained_ML(Model, X_train, y_train, X_test)

            Model_name_technique = "{}_{}".format(Model_name, self.__Enhancement_technique)

            # * 
            #Dir_name = str(Class_problem_prefix) + 'Model_s' + str(Enhancement_technique) + '_dir'
            Dir_name_csv = "{}_Folder_Data_Models_ML_{}".format(self.__Class_problem_prefix, self.__Enhancement_technique)
            Dir_name_images = "{}_Folder_Images_Models_ML_{}".format(self.__Class_problem_prefix, self.__Enhancement_technique)
                
            Dir_name_csv_model = "{}_Folder_Data_Model_ML_{}_{}".format(self.__Class_problem_prefix, Model_name_letters, self.__Enhancement_technique)
            Dir_name_images_model = "{}_Folder_Images_Model_ML_{}_{}".format(self.__Class_problem_prefix, Model_name_letters, self.__Enhancement_technique)
            #print(Folder_CSV + '/' + Dir_name)
            #print('\n')

            # *
            Dir_data_csv = '{}/{}'.format(self.__Folder, Dir_name_csv)
            Dir_data_images = '{}/{}'.format(self.__Folder, Dir_name_images)

            Exist_dir_csv = os.path.isdir(Dir_data_csv)
            Exist_dir_images = os.path.isdir(Dir_data_images)

            # *
            if Exist_dir_csv == False:
                Folder_path = os.path.join(self.__Folder, Dir_name_csv)
                os.mkdir(Folder_path)
                print(Folder_path)
            else:
                Folder_path = os.path.join(self.__Folder, Dir_name_csv)
                print(Folder_path)

            if Exist_dir_images == False:
                Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                os.mkdir(Folder_path_images)
                print(Folder_path_images)
            else:
                Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
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

            # * Save dataframe in the folder given
            #Dataframe_save_name = 'Biclass' + '_Dataframe_' + 'FOF_' + str(Enhancement_technique)  + '.csv'
            Dataframe_save_name = "{}_Dataframe_Folder_Data_Models_{}.csv".format(self.__Class_problem_prefix, self.__Enhancement_technique)
            Dataframe_save_folder = os.path.join(Folder_path_in, Dataframe_save_name)

            # *
            #Confusion_matrix_dataframe_name = 'Dataframe_' + str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.csv'
            Confusion_matrix_dataframe_name = "Dataframe_Confusion_Matrix_{}_{}_{}.csv".format(self.__Class_problem_prefix, Model_name_letters, self.__Enhancement_technique)
            Confusion_matrix_dataframe_folder = os.path.join(Folder_path_in, Confusion_matrix_dataframe_name)
            
            # * 
            Dataframe_ROC_name = "{}_Dataframe_ROC_Curve_Values_{}_{}.csv".format(self.__Class_problem_prefix, Model_name_letters, self.__Enhancement_technique)
            Dataframe_ROC_folder = os.path.join(Folder_path_in, Dataframe_ROC_name)

            # * Save this figure in the folder given
            #Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
            Class_problem_name = "{}_{}_{}.png".format(self.__Class_problem_prefix, Model_name_letters, self.__Enhancement_technique)
            Class_problem_folder = os.path.join(Folder_path_images_in, Class_problem_name)

            # * Create dataframe and define the headers
            Column_names_ = [ 'name model', "model used", "accuracy training FE", "accuracy training LE", 
                            "accuracy testing", "loss train", "loss test", "training images", "validation images", 
                            "test images", "time training", "time testing", "technique used", "TN", "FP", "FN", "TP", "epochs", 
                            "precision", "recall", "f1_Score"]

            Dataframe_save = pd.DataFrame(columns = Column_names_)

            if self.__Classes == 2:

                # * Confusion Matrix
                print('Confusion Matrix')
                Confusion_matrix = confusion_matrix(y_test, Y_pred)
                #cf_matrix = confusion_matrix(y_test, y_pred)

                print(Confusion_matrix)
                print(classification_report(y_test, Y_pred, target_names = self.__Class_labels))

                Dict = classification_report(y_test, Y_pred, target_names = self.__Class_labels, output_dict = True)

                for i, Report_labels in enumerate(Classification_report_labels):
                    for _, Metric_labels in enumerate(Classification_report_metrics_labels):
                        Classification_report_names.append('{} {}'.format(Metric_labels, Report_labels))
                        Classification_report_values.append(Dict[Report_labels][Metric_labels])
                        #print(Dict[Report_labels][Metric_labels])
                print("\n")
                
                # *
                Column_names_.extend(self.__Class_labels)
                Column_names_.extend(Classification_report_names)

                # *
                Dataframe_save = pd.DataFrame(columns = Column_names_)

                # * Accuracy
                Accuracy = accuracy_score(y_test, Y_pred)
                print(f"Accuracy: {round(Accuracy, Digits)}")
                print("\n")

                # * Precision
                Precision = precision_score(y_test, Y_pred)
                print(f"Precision: {round(Precision, Digits)}")
                print("\n")

                # * Recall
                Recall = recall_score(y_test, Y_pred)
                print(f"Recall: {round(Recall, Digits)}")
                print("\n")

                # * F1-score
                F1_score = f1_score(y_test, Y_pred)
                print(f"F1: {round(F1_score, Digits)}")
                print("\n")

                Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))
                Confusion_matrix_dataframe.to_csv(Confusion_matrix_dataframe_folder, index = False)

                # * FPR and TPR values for the ROC curve
                FPR, TPR, _ = roc_curve(y_test, Y_pred)
                Auc = auc(FPR, TPR)
                
                for i in range(len(FPR)):
                    ROC_curve_FPR.append(FPR[i])

                for i in range(len(TPR)):
                    ROC_curve_TPR.append(TPR[i])

                # * Dict_roc_curve

                Dict_roc_curve = {'FPR': ROC_curve_FPR, 'TPR': ROC_curve_TPR} 
                Dataframe_ROC = pd.DataFrame(Dict_roc_curve)
                Dataframe_ROC.to_csv(Dataframe_ROC_folder)

                # *
                Plot_model = FigurePlot(folder = Folder_path_images_in, title = Model_name, 
                                            SI = False, SF = True, CMdf = Confusion_matrix_dataframe_folder, 
                                                ROCdf = [Dataframe_ROC_folder], labels = self.__Class_labels)

                # *
                Plot_model.figure_plot_two()
                Plot_model.figure_plot_ROC_curve()
                Plot_model.figure_plot_CM()

            elif self.__Classes >= 3:
                
                # * Dicts
                FPR = dict()
                TPR = dict()
                Roc_auc = dict()

                Labels_multiclass = []

                # * Binarize labels to get multiples ROC curves
                for i in range(self.__Classes):
                    Labels_multiclass.append(i)
                
                print(Y_pred)

                y_pred_roc = label_binarize(Y_pred, classes = Labels_multiclass)
                y_test_roc = label_binarize(y_test, classes = Labels_multiclass)

                # * Confusion Matrix
                print('Confusion Matrix')
                Confusion_matrix = confusion_matrix(y_test, Y_pred)
                #cf_matrix = confusion_matrix(y_test, y_pred)

                print(confusion_matrix(y_test, Y_pred))
                print(classification_report(y_test, Y_pred, target_names = self.__Class_labels))
                
                Dict = classification_report(y_test, Y_pred, target_names = self.__Class_labels, output_dict = True)

                for i, Report_labels in enumerate(Classification_report_labels):
                    for _, Metric_labels in enumerate(Classification_report_metrics_labels):
                        Classification_report_names.append('{} {}'.format(Metric_labels, Report_labels))
                        Classification_report_values.append(Dict[Report_labels][Metric_labels])
                        #print(Dict[Report_labels][Metric_labels])
                print("\n")

                # *
                Column_names_.extend(self.__Class_labels)
                Column_names_.extend(Classification_report_names)

                # *
                Dataframe_save = pd.DataFrame(columns = Column_names_)

                # * 
                Dataframe_save.to_csv(Dataframe_save_folder)

                # * Accuracy
                Accuracy = accuracy_score(y_test, Y_pred)
                print(f"Precision: {round(Accuracy, Digits)}")
                print("\n")

                # * Precision
                Precision = precision_score(y_test, Y_pred, average = 'weighted')
                print(f"Precision: {round(Precision, Digits)}")
                print("\n")

                # * Recall
                Recall = recall_score(y_test, Y_pred, average = 'weighted')
                print(f"Recall: {round(Recall, Digits)}")
                print("\n")

                # * F1-score
                F1_score = f1_score(y_test, Y_pred, average = 'weighted')
                print(f"F1: {round(F1_score, Digits)}")
                print("\n")

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

                    Dataframe_ROC_name = "{}_Dataframe_ROC_Curve_Values_{}_{}_{}.csv".format(self.__Class_problem_prefix, Model_name, self.__Enhancement_technique, j)
                    Dataframe_ROC_folder = os.path.join(Folder_path_in, Dataframe_ROC_name)
                    Dataframe_ROC.to_csv(Dataframe_ROC_folder)
                    Dataframe_ROCs.append(Dataframe_ROC_folder)

                # *
                Plot_model = FigurePlot(folder = Folder_path_images_in, title = Model_name, 
                                            SI = False, SF = True, CMdf = Confusion_matrix_dataframe_folder, 
                                                ROCdf = [i for i in Dataframe_ROCs], labels = self.__Class_labels)

                # *
                Plot_model.figure_plot_two_multiclass()
                Plot_model.figure_plot_ROC_curve_multiclass()
                Plot_model.figure_plot_CM()

            Info.append(Model_name_technique)
            Info.append(Model_name)

            Info.append(len(y_train))
            Info.append(len(y_test))
            Info.append(Total_time_training)
            Info.append(self.__Enhancement_technique)

            Info.append(Confusion_matrix[0][0])
            Info.append(Confusion_matrix[0][1])
            Info.append(Confusion_matrix[1][0])
            Info.append(Confusion_matrix[1][1])

            Info.append(self.__Epochs)
            Info.append(Accuracy)
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

            self.overwrite_dic_CSV_folder(Dataframe_save, Dataframe_save_folder, Column_names_, Info)
