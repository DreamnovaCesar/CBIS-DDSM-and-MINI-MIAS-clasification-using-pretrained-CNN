from Final_Code_0_0_Libraries import *

#Dataframe, Dataframe_save, Folder_path, Column_names, ML_model, Enhancement_technique, Extract_feature_technique, Class_labels, Folder_data, Folder_models

def machine_learning_config(**kwargs):

    """
	  Extract features from each image, it could be FOS, GLCM or GRLM.

    Parameters:
    argument1 (dataframe): Datraframe without values.
    argument2 (list): Values of each model
    argument3 (folder): Folder to save images with metadata
    argument4 (folder): Folder to save images with metadata (CSV)
    argument5 (str): Name of each model

    Returns:
	  list:Returning all metadata of each model trained.
    
   	"""

    # * General attributes
    Dataframe = kwargs.get('dataframe', None)
    Dataframe_save = kwargs.get('dataframesave', None)

    # *
    Folder_path = kwargs.get('folderpath', None)
    Folder_CSV = kwargs.get('foldercsv', None)
    Folder_models = kwargs.get('foldermodels', None)
    Folder_models_esp = kwargs.get('foldermodelsesp', None)

    # *
    Models = kwargs.get('models', None)

    # *
    Extract_feature_technique = kwargs.get('MLT', None)
    Enhancement_technique = kwargs.get('technique', None)
    Class_labels = kwargs.get('labels', None)
    Column_names = kwargs.get('columns', None)

    sm = SMOTE()
    sc = StandardScaler()
    #ALL_ML_model = len(ML_model)

    # * Class problem definition
    Class_problem = len(Class_labels)

    if Class_problem == 2:
        Class_problem_prefix = 'Biclass'
    elif Class_problem >= 3:
        Class_problem_prefix = 'Multiclass'

    for Index, Model in enumerate(Models):

        # * Extract data and label
        Dataframe_len_columns = len(Dataframe.columns)

        X_total = Dataframe.iloc[:, 0:Dataframe_len_columns - 1].values
        Y_total = Dataframe.iloc[:, -1].values

        #print(X_total)
        #print(Y_total)

        #pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
        #print(Dataframe)

        # * Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(np.array(X_total), np.array(Y_total), test_size = 0.2, random_state = 1)

        # * Resample data for training
        X_train, y_train = sm.fit_resample(X_train, y_train)

        # * Scaling data for training
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #print(y_train)
        #print(y_test)

        # * Chose the machine learning.
        Info_model = machine_learning_models(Model, Enhancement_technique, Class_labels, X_train, y_train, X_test, y_test, Folder_models)
    
        Dataframe_results = overwrite_row_CSV(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    # * Save dataframe in the folder given
    #Class_problem_dataframe = str(Class_problem_prefix) + 'Dataframe_' + str(Extract_feature_technique) + '_' + str(Enhancement_technique) + '.csv'
    Class_problem_dataframe = "{}_Dataframe_{}_{}.csv".format(Class_problem_prefix, Extract_feature_technique, Enhancement_technique)
    Class_problem_folder = os.path.join(Folder_CSV, Class_problem_dataframe)
    Dataframe.to_csv(Class_problem_folder)

    return Dataframe_results

# ?
#ML_model, Enhancement_technique, Class_labels, X_train, y_train, X_test, y_test, Folder_models, Dataframe_save, Folder_path, Info_model, Column_names, Index
def machine_learning_models(**kwargs):

    """
	  General configuration for each model, extracting features and printing theirs values (Machine Learning).

    Parameters:
    argument1 (model): Model chosen.
    argument2 (str): technique used.
    argument3 (list): labels used for printing.
    argument4 (int): X train split data.
    argument5 (int): y train split data.
    argument6 (int): X test split data.
    argument7 (int): y test split data.
    argument8 (int): Folder used to save data images.
    argument9 (int): Folder used to save data images in spanish.

    Returns:
	  int:Returning all data from each model.
    
   	"""

    # * General attributes
    #Dataframe = kwargs.get('dataframe', None)
    Dataframe_save = kwargs.get('dataframesave', None)

    # *
    Folder_path = kwargs.get('folderpath', None)
    Folder_CSV = kwargs.get('foldercsv', None)
    Folder_models = kwargs.get('foldermodels', None)
    Folder_models_esp = kwargs.get('foldermodelsesp', None)

    # *
    Model = kwargs.get('model', None)

    # *
    Enhancement_technique = kwargs.get('technique', None)
    Class_labels = kwargs.get('labels', None)
    Column_names = kwargs.get('columns', None)

    # *
    X_train = kwargs.get('Xtrain', None)
    y_train = kwargs.get('ytrain', None)

    X_test = kwargs.get('Xtest', None)
    y_test = kwargs.get('ytest', None)

    # *
    Index = kwargs.get('index', None)

    # * Parameters plt

    Height = 5
    Width = 12
    Annot_kws = 12
    Font = 0.7
    H = 0.02
    
    X_size_figure = 1
    Y_size_figure = 2
    
    # * Parameters dic classification report

    Macro_avg_label = 'macro avg'
    Weighted_avg_label = 'weighted avg'

    Classification_report_labels = []
    Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

    for Label in Class_labels:
      Classification_report_labels.append(Label)
    
    Classification_report_labels.append(Macro_avg_label)
    Classification_report_labels.append(Weighted_avg_label)

    Classification_report_values = []

    # * Metrics digits

    Digits = 4

    # * List
    Info = []
    ROC_curve_FPR = []
    ROC_curve_TPR = []
    Labels_multiclass = []
    
    # * Class problem definition
    Class_problem = len(Class_labels)

    # * Conditional if the class problem was biclass or multiclass

    if Class_problem == 2:
        Class_problem_prefix = 'Biclass'
    elif Class_problem > 2:
        Class_problem_prefix = 'Multiclass'

    # * Lists
    Column_names_ = [ 'name model', "model used", "accuracy training FE", "accuracy training LE", 
                      "accuracy testing", "loss train", "loss test", "training images", "validation images", 
                      "test images", "time training", "time testing", "technique used", "TN", "FP", "FN", "TP", "epochs", 
                      "precision", "recall", "f1_Score"]

    Dataframe_save = pd.DataFrame(columns = Column_names_)

    # * Save dataframe in the folder given
    #Dataframe_save_name = 'Biclass' + '_Dataframe_' + 'FOF_' + str(Enhancement_technique)  + '.csv'
    Dataframe_save_name = "{}_Dataframe_Folder_Data_Models_{}.csv".format(Class_problem_prefix, Enhancement_technique)
    Dataframe_save_folder = os.path.join(Folder_CSV, Dataframe_save_name)

    # *
    #Confusion_matrix_dataframe_name = 'Dataframe_' + str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.csv'
    Confusion_matrix_dataframe_name = "Dataframe_Confusion_Matrix_{}_{}_{}.csv".format(Class_problem_prefix, Pretrained_model_name, Enhancement_technique)
    Confusion_matrix_dataframe_folder = os.path.join(Folder_path_in, Confusion_matrix_dataframe_name)

    # * 
    #Dir_name = str(Class_problem_prefix) + 'Model_s' + str(Enhancement_technique) + '_dir'
    Dir_name_csv = "{}_Folder_Data_Models_{}".format(Class_problem_prefix, Enhancement_technique)
    Dir_name_images = "{}_Folder_Images_Models_{}".format(Class_problem_prefix, Enhancement_technique)
        
    Dir_name_csv_model = "{}_Folder_Data_Model_{}_{}".format(Class_problem_prefix, Pretrained_model_name_letters, Enhancement_technique)
    Dir_name_images_model = "{}_Folder_Images_Model_{}_{}".format(Class_problem_prefix, Pretrained_model_name_letters, Enhancement_technique)
    #print(Folder_CSV + '/' + Dir_name)
    #print('\n')

    # * 
    Dataframe_ROC_name = "{}_Dataframe_ROC_Curve_Values_{}_{}.csv".format(Class_problem_prefix, Pretrained_model_name, Enhancement_technique)
    Dataframe_ROC_folder = os.path.join(Folder_path_in, Dataframe_ROC_name)

    # * Save this figure in the folder given
    #Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
    Class_problem_name = "{}_{}_{}.png".format(Class_problem_prefix, Pretrained_model_name, Enhancement_technique)
    Class_problem_folder = os.path.join(Folder_path_images_in, Class_problem_name)

    # *
    Dir_data_csv = Folder_CSV + '/' + Dir_name_csv
    Dir_data_images = Folder_CSV + '/' + Dir_name_images

    Exist_dir_csv = os.path.isdir(Dir_data_csv)
    Exist_dir_images = os.path.isdir(Dir_data_images)

    # *
    if Exist_dir_csv == False:
        Folder_path = os.path.join(Folder_CSV, Dir_name_csv)
        os.mkdir(Folder_path)
        print(Folder_path)
    else:
        Folder_path = os.path.join(Folder_CSV, Dir_name_csv)
        print(Folder_path)

    if Exist_dir_images == False:
        Folder_path_images = os.path.join(Folder_CSV, Dir_name_images)
        os.mkdir(Folder_path_images)
        print(Folder_path_images)
    else:
        Folder_path_images = os.path.join(Folder_CSV, Dir_name_images)
        print(Folder_path_images)

    Dir_data_csv_model = Folder_path + '/' + Dir_name_csv_model
    Dir_data_images_model = Folder_path_images + '/' + Dir_name_images_model
    
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

    # *
    Dataframe_save.to_csv(Dataframe_save_folder)
    #Dataframe_save = pd.read_csv(Dataframe_save_folder)

    if len(Class_labels) == 2:

        # * Get the data from the model chosen
        Y_pred, Model_name, Model_name_letters, Total_time_training = ML_model_pretrained(0, X_train, y_train, X_test)

        # * Confusion Matrix
        Confusion_matrix = confusion_matrix(y_test, Y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(Confusion_matrix)
        print(classification_report(y_test, Y_pred, target_names = Class_labels))

        Dict = classification_report(y_test, Y_pred, target_names = Class_labels, output_dict = True)

        for i, Report_labels in enumerate(Classification_report_labels):
            for _, Metric_labels in enumerate(Classification_report_metrics_labels):
                print(Dict[Report_labels][Metric_labels])
                Classification_report_values.append(Dict[Report_labels][Metric_labels])
                print("\n")
                
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
        F1_Score = f1_score(y_test, Y_pred)
        print(f"F1: {round(F1_Score, Digits)}")
        print("\n")

        Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))
        Confusion_matrix_dataframe.to_csv(Confusion_matrix_dataframe_folder, index = False)

        # * Figure's size
        plt.figure(figsize = (Width, Height))
        plt.subplot(X_size_figure, Y_size_figure, 1)
        sns.set(font_scale = Font) # for label size

        # * Confusion matrix heatmap
        ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(Class_labels)
        ax.set_yticklabels(Class_labels)

        # * FPR and TPR values for the ROC curve
        FPR, TPR, _ = roc_curve(y_test, Y_pred)
        Auc = auc(FPR, TPR)
        
        # * Subplot ROC curve
        plt.subplot(X_size_figure, Y_size_figure, 2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(FPR, TPR, label = Model_name + '(area = {:.4f})'.format(Auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        # * Save this figure in the folder given
        #Class_problem_name = Class_problem_prefix + str(Model_name) + str(Enhancement_technique) + '_' + '.png'
        Class_problem_name = "{}_{}_{}.csv".format(Class_problem_prefix, Model_name, Enhancement_technique)
        Class_problem_folder = os.path.join(Folder_models, Class_problem_name)
        
        for i in range(len(FPR)):
            ROC_curve_FPR.append(FPR[i])

        for i in range(len(TPR)):
            ROC_curve_TPR.append(TPR[i])

        # * Dict_roc_curve

        Dict_roc_curve = {'FPR': ROC_curve_FPR, 'TPR': ROC_curve_TPR} 
        Dataframe_ROC = pd.DataFrame(Dict_roc_curve)

        Dataframe_ROC.to_csv(Dataframe_ROC_folder)

        plt.savefig(Class_problem_folder)
        #plt.show()

    elif len(Class_labels) >= 3:

        # * Get the data from the model chosen
        Y_pred, Model_name, Model_name_letters, Total_time_training = ML_model_pretrained(0, X_train, y_train, X_test)

        # * Binarize labels to get multiples ROC curves
        for i in range(len(Class_labels)):
            Labels_multiclass.append(i)
        
        print(Y_pred)

        y_pred_roc = label_binarize(Y_pred, classes = Labels_multiclass)
        y_test_roc = label_binarize(y_test, classes = Labels_multiclass)

        # * Confusion Matrix
        Confusion_matrix = confusion_matrix(y_test, Y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(confusion_matrix(y_test, Y_pred))
        print(classification_report(y_test, Y_pred, target_names = Class_labels))
        
        Dict = classification_report(y_test, Y_pred, target_names = Class_labels, output_dict = True)

        for i, Report_labels in enumerate(Classification_report_labels):
            for j, Metric_labels in enumerate(Classification_report_metrics_labels):
                print(Dict[Report_labels][Metric_labels])
                Classification_report_values.append(Dict[Report_labels][Metric_labels])
                print("\n")

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
        F1_Score = f1_score(y_test, Y_pred, average = 'weighted')
        print(f"F1: {round(F1_Score, Digits)}")
        print("\n")

        Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))
        Confusion_matrix_dataframe.to_csv(Confusion_matrix_dataframe_folder, index = False)

        # * Figure's size
        plt.figure(figsize = (Width, Height))
        plt.subplot(X_size_figure, Y_size_figure, 1)
        sns.set(font_scale = Font) # for label size

        # * Confusion matrix heatmap
        ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd') # font size
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(Class_labels)
        ax.set_yticklabels(Class_labels)

        # * FPR and TPR values for the ROC curve
        FPR = dict()
        TPR = dict()
        Roc_auc = dict()

        for i in range(Class_problem):
            FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
            Roc_auc[i] = auc(FPR[i], TPR[i])

        colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
        
        # * Subplot several ROC curves
        plt.subplot(X_size_figure, Y_size_figure, 2)
        plt.plot([0, 1], [0, 1], 'k--')

        for i, color, lbl in zip(range(Class_problem), colors, Class_labels):
            plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        # * Save this figure in the folder given
        #Class_problem_name = Class_problem_prefix + str(Model_name) + str(Enhancement_technique) + '.png'
        Class_problem_name = "{}_{}_{}.csv".format(Class_problem_prefix, Model_name, Enhancement_technique)
        Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

        plt.savefig(Class_problem_folder)
        #plt.show()
    
    Info.append(Model_name + '_' + Enhancement_technique)
    Info.append(Model_name)
    Info.append(Accuracy)
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_Score)

    for i, value in enumerate(Classification_report_values):
      Info.append(value)

    Info.append(len(y_train))
    Info.append(len(y_test))
    Info.append(Total_time_training)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])

    if Class_problem == 2:
        Info.append(Auc)
    elif Class_problem >= 3:
        for i in range(Class_problem):
            Info.append(Roc_auc[i])

    Dataframe_updated = overwrite_row_CSV(Dataframe_save, Folder_path, Info, Column_names, Index)

    return Info

# ? Folder Update CSV changing value

def overwrite_row_CSV(Folder_path, Dataframe, Info, Column_names, Row):

    """
	  Updates final CSV dataframe to see all values

    Parameters:
    argument1 (list): All values.
    argument2 (dataframe): dataframe that will be updated
    argument3 (list): Names of each column
    argument4 (folder): Folder path to save the dataframe
    argument5 (int): The index.

    Returns:
	  void
    
   	"""

    for i in range(len(Info)):
        Dataframe.loc[Row, Column_names[i]] = Info[i]
  
    Dataframe.to_csv(Folder_path, index = False)
  
    print(Dataframe)

    return Dataframe

def ML_model_pretrained(Model_pretrained_value: int, X_train, y_train, X_test):
    """
    Model configuration.

    Model_pretrained_value is a import variable that chose the model required.
    The next index show every model this function has:

    1: Support Vector Machine
    2: Multi Support Vector Machine
    3: Decision Tree
    4: K Neighbors
    5: Random Forest
    6: Gradient Boostin Classifier

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

    # * Function to show the pretrained model.
    def model_pretrained_print(Model_name, Model_name_letters):
        print('The model chosen is: {} -------- {} ✅'.format(Model_name, Model_name_letters))

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

    model_pretrained_print(Model_name, Model_name_letters)

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = Model_index_chosen(kernel = 'rbf', C = 1)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time

    Y_pred = classifier.predict(X_test)

    return Y_pred, Model_name, Model_name_letters, Total_time_training
