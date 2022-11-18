
from Final_Code_0_0_Libraries import *
from Final_Code_0_1_Class_Utilities import Utilities

# ? Figure Adjust

class FigureAdjust(Utilities):
    """
    Utilities inheritance.

    A class to set up the initial values of each statistic plot

    Methods:
        data_dic(): description

        show_figure(): description

        save_figure(): description
        
    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Keyword Args:
            folder (str): description 
            title (str): description
            SI (bool): description
            SF (bool): description
            classes (int): description
        """

        # *
        self._Folder_path = kwargs.get('folder', None)
        self._Title = kwargs.get('title', None)

        # * 
        self._Show_image = kwargs.get('SI', False)
        self._Save_figure = kwargs.get('SF', False)

        # *
        self._Num_classes = kwargs.get('classes', None)

        # *
        self._X_figure_size = 12
        self._Y_figure_size = 12

        # * General parameters
        self._Font_size_title = self._X_figure_size * 0.1
        self._Font_size_general = self._X_figure_size * 0.1
        self._Font_size_confusion = self._X_figure_size * 0.8
        self._Font_size_ticks = self._X_figure_size  * 0.1

        # * 
        #self.Annot_kws = kwargs.get('annot_kws', None)
        #self.Font = kwargs.get('font', None)
    
    # * Class variables
    def __repr__(self):
            return f'[{self._Folder_path}, {self._Title}, {self._Show_image}, {self._Save_figure}, {self._Num_classes}, {self._X_figure_size}, {self._Y_figure_size}]';

    # * Class description
    def __str__(self):
        return  f'A class to set up the initial values of each statistic plot';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, figure adjust destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self._Folder_path),
                'Title': str(self._Title),
                'Show image': str(self._Show_image),
                'Save figure': str(self._Save_figure),
                'Number of classes': str(self._Num_classes),
                "figure's height": str(self._X_figure_size),
                "figure's width": str(self._Y_figure_size),
                };

    # * _Folder_path attribute
    @property
    def _Folder_path_property(self):
        return self._Folder_path

    @_Folder_path_property.setter
    def _Folder_path_property(self, New_value):
        self._Folder_path = New_value
    
    @_Folder_path_property.deleter
    def _Folder_path_property(self):
        print("Deleting Folder_path...")
        del self._Folder_path

    # * Title attribute
    @property
    def _Title_property(self):
        return self._Title

    @_Title_property.setter
    def _Title_property(self, New_value):
        self._Title = New_value
    
    @_Title_property.deleter
    def _Title_property(self):
        print("Deleting Title...")
        del self._Title

    # * Show_image attribute
    @property
    def _Show_image_property(self):
        return self._Show_image

    @_Show_image_property.setter
    def _Show_image_property(self, New_value):
        self._Show_image = New_value
    
    @_Show_image_property.deleter
    def _Show_image_property(self):
        print("Deleting Show_image...")
        del self._Show_image

    # * Save_figure attribute
    @property
    def _Save_figure_property(self):
        return self._Save_figure

    @_Save_figure_property.setter
    def _Save_figure_property(self, New_value):
        self._Save_figure = New_value
    
    @_Save_figure_property.deleter
    def _Save_figure_property(self):
        print("Deleting Save_figure...")
        del self._Save_figure

    # * Num_classes attribute
    @property
    def _Num_classes_property(self):
        return self._Num_classes

    @_Num_classes_property.setter
    def _Num_classes_property(self, New_value):
        self._Num_classes = New_value
    
    @_Num_classes_property.deleter
    def _Num_classes_property(self):
        print("Deleting Num_classes...")
        del self._Num_classes

    # ? Static method to show the image.
    @staticmethod
    def show_figure(Show_image: bool = False) -> None:

        if(Show_image == True):
            plt.show()
        
        else: 
            pass

    # ? Static method to save the figure.
    @staticmethod
    def save_figure(Save_figure: bool, Title: int, Func_: str, Folder: str) -> None:

        if(Save_figure == True):
            
            Figure_name = 'Figure_{}_{}.png'.format(Title, Func_)
            Figure_folder = os.path.join(Folder, Figure_name)
            plt.savefig(Figure_folder)

        else:
            pass
    
# ?
class BarChart(FigureAdjust):
    """
    FigureAdjust inheritance.

    A class to set up the barchart using a dataframe. 

    Methods:
        data_dic(): description

        barchart_horizontal(): description

        barchart_vertical(): description
        
    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Keyword Args:
            folder (str): description 
            title (str): description
            SI (bool): description
            SF (bool): description
            classes (int): description
            csv (str): description
            label (str): description
            column (int): description
            reverse (bool): description
        """

        # *
        self._CSV_path = kwargs.get('csv', None)

        # *
        self._Plot_x_label = kwargs.get('label', None)
        self._Plot_column = kwargs.get('column', None)
        self._Plot_reverse = kwargs.get('reverse', None)

        # * Read dataframe csv
        self._Dataframe = pd.read_csv(self._CSV_path)

        # *
        self._Colors = ('gray', 'red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')
        
        # * General lists
        self._X_fast_list_values = []
        self._X_slow_list_values = []

        self._Y_fast_list_values = []
        self._Y_slow_list_values = []

        self._X_fastest_list_value = []
        self._Y_fastest_list_value = []

        self._X_slowest_list_value = []
        self._Y_slowest_list_value = []

        # * Chosing label
        if self._Num_classes == 2:
            self._Label_class_name = 'Biclass'
        elif self._Num_classes > 2:
            self._Label_class_name = 'Multiclass'

    # * Class variables
    def __repr__(self):
            return f'[{self._Folder_path}, {self._Title}, {self._Show_image}, {self._Save_figure}, {self._Num_classes}, {self._X_figure_size}, {self._Y_figure_size}, {self._CSV_path}, {self._Plot_x_label}, {self._Plot_column}, {self._Plot_reverse}]';

    # * Class description
    def __str__(self):
        return  f'A class to set up the barchart using a dataframe';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, barchart destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self._Folder_path),
                'Title': str(self._Title),
                'Show image': str(self._Show_image),
                'Save figure': str(self._Save_figure),
                'Number of classes': str(self._Num_classes),
                "figure's height": str(self._X_figure_size),
                "figure's width": str(self._Y_figure_size),
                'Folder CSV path': str(self._CSV_path),
                'Plot_x_label': str(self._Plot_x_label),
                "Plot_column": str(self._Plot_column),
                "Plot_reverse": str(self._Plot_reverse),
                };

    # * _CSV_path attribute
    @property
    def _CSV_path_property(self):
        return self._CSV_path

    @_CSV_path_property.setter
    def _CSV_path_property(self, New_value):
        self._CSV_path = New_value
    
    @_CSV_path_property.deleter
    def _CSV_path_property(self):
        print("Deleting CSV_path...")
        del self._CSV_path

    # * Plot_x_label attribute
    @property
    def _Plot_x_label_property(self):
        return self._Plot_x_label

    @_Plot_x_label_property.setter
    def _Plot_x_label_property(self, New_value):
        self._Plot_x_label = New_value
    
    @_Plot_x_label_property.deleter
    def _Plot_x_label_property(self):
        print("Deleting Plot_x_label...")
        del self._Plot_x_label

    # * Plot_column attribute
    @property
    def _Plot_column_property(self):
        return self._Plot_column

    @_Plot_column_property.setter
    def _Plot_column_property(self, New_value):
        self._Plot_column = New_value
    
    @_Plot_column_property.deleter
    def _Plot_column_property(self):
        print("Deleting Plot_column...")
        del self._Plot_column

    # * Plot_reverse attribute
    @property
    def _Plot_reverse_property(self):
        return self._Plot_reverse

    @_Plot_reverse_property.setter
    def _Plot_reverse_property(self, New_value):
        self._Plot_reverse = New_value
    
    @_Plot_reverse_property.deleter
    def _Plot_reverse_property(self):
        print("Deleting Plot_reverse...")
        del self._Plot_reverse

    # * _Name attribute
    @property
    def _Name_property(self):
        return self._Name

    @_Name_property.setter
    def _Name_property(self, New_value):
        self._Name = New_value
    
    @_Name_property.deleter
    def _Name_property(self):
        print("Deleting Name...")
        del self._Name

    # ? Method to plot horizontal barcharts.
    @Utilities.timer_func
    def barchart_horizontal(self) -> None:
        """
        Method to plot horizontal barcharts.

        """

        # *
        Horizontal = "horizontal"

        # Initialize the lists for X and Y
        #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

        # * Get X and Y values
        X = list(self._Dataframe.iloc[:, 1])
        Y = list(self._Dataframe.iloc[:, self._Plot_column])

        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))

        # * Reverse is a bool variable with the postion of the plot
        if self._Plot_reverse == True:

            for Index, (i, k) in enumerate(zip(X, Y)):
                if k < np.mean(Y):
                    self._X_fast_list_values.append(i)
                    self._Y_fast_list_values.append(k)
                elif k >= np.mean(Y):
                    self._X_slow_list_values.append(i)
                    self._Y_slow_list_values.append(k)

            for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
                if k == np.min(self._Y_fast_list_values):
                    self._X_fastest_list_value.append(i)
                    self._Y_fastest_list_value.append(k)
                    #print(X_fastest_list_value)
                    #print(Y_fastest_list_value)

            for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
                if k == np.max(self._Y_slow_list_values):
                    self._X_slowest_list_value.append(i)
                    self._Y_slowest_list_value.append(k)

        else:

            for Index, (i, k) in enumerate(zip(X, Y)):
                if k < np.mean(Y):
                    self._X_slow_list_values.append(i)
                    self._Y_slow_list_values.append(k)
                elif k >= np.mean(Y):
                    self._X_fast_list_values.append(i)
                    self._Y_fast_list_values.append(k)

            for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
                if k == np.max(self._Y_fast_list_values):
                    self._X_fastest_list_value.append(i)
                    self._Y_fastest_list_value.append(k)
                    #print(XFastest)
                    #print(YFastest)

            for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
                if k == np.min(self._Y_slow_list_values):
                    self._X_slowest_list_value.append(i)
                    self._Y_slowest_list_value.append(k)

        # * Plot the data using bar() method
        plt.bar(self._X_slow_list_values, self._Y_slow_list_values, label = "Bad", color = 'gray')
        plt.bar(self._X_slowest_list_value, self._Y_slowest_list_value, label = "Worse", color = 'black')
        plt.bar(self._X_fast_list_values, self._Y_fast_list_values, label = "Better", color = 'lightcoral')
        plt.bar(self._X_fastest_list_value, self._Y_fastest_list_value, label = "Best", color = 'red')

        # *
        for Index, value in enumerate(self._Y_slowest_list_value):
            plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self._X_slowest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

        # *
        for Index, value in enumerate(self._Y_fastest_list_value):
            plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self._X_fastest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

        plt.legend(fontsize = self._Font_size_general)

        plt.title(self._Title, fontsize = self._Font_size_title)
        plt.xlabel(self._Plot_x_label, fontsize = self._Font_size_general)
        plt.xticks(fontsize = self._Font_size_ticks)
        plt.ylabel("Models", fontsize = self._Font_size_general)
        plt.yticks(fontsize = self._Font_size_ticks)
        plt.grid(color = self._Colors[0], linestyle = '-', linewidth = 0.2)

        # *
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        xmin, xmax = axes.get_xlim()

        # *
        for i, value in enumerate(self._Y_slow_list_values):
            plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

            Next_value = i

        Next_value = Next_value + 1

        for i, value in enumerate(self._Y_fast_list_values):
            plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

        #plt.savefig(Graph_name_folder)

        self.save_figure(self._Save_figure, self._Title, Horizontal, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot Vertical barcharts.
    @Utilities.timer_func
    def barchart_vertical(self) -> None:  
        """
        Method to plot Vertical barcharts.
        """

        # *
        Vertical = "Vertical"

        # Initialize the lists for X and Y
        #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

        # * Get X and Y values
        X = list(self._Dataframe.iloc[:, 1])
        Y = list(self._Dataframe.iloc[:, self._Plot_column])

        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))

        # * Reverse is a bool variable with the postion of the plot
        if self._Plot_reverse == True:

            for Index, (i, k) in enumerate(zip(X, Y)):
                if k < np.mean(Y):
                    self._X_fast_list_values.append(i)
                    self._Y_fast_list_values.append(k)
                elif k >= np.mean(Y):
                    self._X_slow_list_values.append(i)
                    self._Y_slow_list_values.append(k)

            for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
                if k == np.min(self._Y_fast_list_values):
                    self._X_fastest_list_value.append(i)
                    self._Y_fastest_list_value.append(k)
                    #print(X_fastest_list_value)
                    #print(Y_fastest_list_value)

            for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
                if k == np.max(self._Y_slow_list_values):
                    self._X_slowest_list_value.append(i)
                    self._Y_slowest_list_value.append(k)

        else:

            for Index, (i, k) in enumerate(zip(X, Y)):
                if k < np.mean(Y):
                    self._X_slow_list_values.append(i)
                    self._Y_slow_list_values.append(k)
                elif k >= np.mean(Y):
                    self._X_fast_list_values.append(i)
                    self._Y_fast_list_values.append(k)

            for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
                if k == np.max(self._Y_fast_list_values):
                    self._X_fastest_list_value.append(i)
                    self._Y_fastest_list_value.append(k)
                    #print(XFastest)
                    #print(YFastest)

            for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
                if k == np.min(self._Y_slow_list_values):
                    self._X_slowest_list_value.append(i)
                    self._Y_slowest_list_value.append(k)

        # * Plot the data using bar() method
        plt.bar(self._X_slow_list_values, self._Y_slow_list_values, label = "Bad", color = 'gray')
        plt.bar(self._X_slowest_list_value, self._Y_slowest_list_value, label = "Worse", color = 'black')
        plt.bar(self._X_fast_list_values, self._Y_fast_list_values, label = "Better", color = 'lightcoral')
        plt.bar(self._X_fastest_list_value, self._Y_fastest_list_value, label = "Best", color = 'red')

        # *
        for Index, value in enumerate(self._Y_slowest_list_value):
            plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self._X_slowest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

        # *
        for Index, value in enumerate(self._Y_fastest_list_value):
            plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self._X_fastest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

        plt.legend(fontsize = self._Font_size_general)

        plt.title(self._Title, fontsize = self._Font_size_title)
        plt.xlabel(self._Plot_x_label, fontsize = self._Font_size_general)
        plt.xticks(fontsize = self._Font_size_ticks)
        plt.ylabel("Models", fontsize = self._Font_size_general)
        plt.yticks(fontsize = self._Font_size_ticks)
        plt.grid(color = self._Colors[0], linestyle = '-', linewidth = 0.2)

        # *
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        xmin, xmax = axes.get_xlim()

        # *
        for i, value in enumerate(self._Y_slow_list_values):
            plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

            Next_value = i

        Next_value = Next_value + 1

        for i, value in enumerate(self._Y_fast_list_values):
            plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

        #plt.savefig(Graph_name_folder)

        self.save_figure(self._Save_figure, self._Title, Vertical, self._Folder_path)
        self.show_figure(self._Show_image)

# ? Create class folders
class FigurePlot(FigureAdjust):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # * 
        self._Labels = kwargs.get('labels', None)

        # * 
        self._CM_dataframe = kwargs.get('CMdf', None)
        self._History_dataframe = kwargs.get('Hdf', None)
        self._ROC_dataframe = kwargs.get('ROCdf', None)

        # *
        self._X_size_figure_subplot = 2
        self._Y_size_figure_subplot = 2

        self._Roc_curve_dataframes = []


        # *
        if isinstance(self._CM_dataframe, str):
            self._Confusion_matrix_dataframe = pd.read_csv(self._CM_dataframe)
        
        if isinstance(self._History_dataframe, str):
            self._History_data_dataframe = pd.read_csv(self._History_dataframe)

            self._Accuracy = self._History_data_dataframe.accuracy.to_list()
            self._Loss = self._History_data_dataframe.loss.to_list()
            self._Val_accuracy = self._History_data_dataframe.val_accuracy.to_list()
            self._Val_loss = self._History_data_dataframe.val_loss.to_list()

        if isinstance(self._ROC_dataframe, list):
            for Dataframe in self._ROC_dataframe:
                self._Roc_curve_dataframes.append(pd.read_csv(Dataframe))

        self._FPRs = []
        self._TPRs = []
        
        for i in range(len(self._Roc_curve_dataframes)):
            self._FPRs.append(self._Roc_curve_dataframes[i].FPR.to_list())
            self._TPRs.append(self._Roc_curve_dataframes[i].TPR.to_list())

    # * Class variables
    def __repr__(self):
            return f'[{self._Folder_path}, {self._Title}, {self._Show_image}, {self._Save_figure}, {self._Num_classes}, {self._X_figure_size}, {self._Y_figure_size}, {self._CSV_path}, {self._Plot_x_label}, {self._Plot_column}, {self._Plot_reverse}]';

    # * Class description
    def __str__(self):
        return  f'A class to set up the barchart using a dataframe';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, barchart destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(self._Folder_path),
                'Title': str(self._Title),
                'Show image': str(self._Show_image),
                'Save figure': str(self._Save_figure),
                'Number of classes': str(self._Num_classes),
                "figure's height": str(self._X_figure_size),
                "figure's width": str(self._Y_figure_size),
                'Folder CSV path': str(self._CSV_path),
                'Plot_x_label': str(self._Plot_x_label),
                "Plot_column": str(self._Plot_column),
                "Plot_reverse": str(self._Plot_reverse),
                };

    # * _CSV_path attribute
    @property
    def _CSV_path_property(self):
        return self._CSV_path

    @_CSV_path_property.setter
    def _CSV_path_property(self, New_value):
        self._CSV_path = New_value
    
    @_CSV_path_property.deleter
    def _CSV_path_property(self):
        print("Deleting CSV_path...")
        del self._CSV_path

    # * _Roc_curve_dataframe attribute
    @property
    def _Roc_curve_dataframe_property(self):
        return self._Roc_curve_dataframe

    @_Roc_curve_dataframe_property.setter
    def _Roc_curve_dataframe_property(self, New_value):
        self._Roc_curve_dataframe = New_value
    
    @_Roc_curve_dataframe_property.deleter
    def _Roc_curve_dataframe_property(self):
        print("Deleting Roc_curve_dataframe...")
        del self._Roc_curve_dataframe

    # ? Method to plot four images in a figure (accuracy, loss, confusion matrix, and ROC curve)
    @Utilities.timer_func
    def figure_plot_four(self) -> None: 
        """_summary_
        """

        # *
        Four_plot = 'Four_plot'

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
        plt.suptitle(self._Title, fontsize = self._Font_size_title)
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 4)

        # * Confusion matrix heatmap
        sns.set(font_scale = self._Font_size_general)

        # *
        ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_confusion})
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        # * Subplot training accuracy
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)
        plt.plot(self._Accuracy, label = 'Training Accuracy')
        plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc = 'lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')

        # * Subplot training loss
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
        plt.plot(self._Loss, label = 'Training Loss')
        plt.plot(self._Val_loss, label = 'Validation Loss')
        plt.ylim([0, 2.0])
        plt.legend(loc = 'upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')

        # * FPR and TPR values for the ROC curve
        Auc = auc(self._FPRs[0], self._TPRs[0])

        # * Subplot ROC curve
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 3)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self._FPRs[0], self._TPRs[0], label = 'Test' + '(area = {:.4f})'.format(Auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')
        
        self.save_figure(self._Save_figure, self._Title, Four_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot four images in a figure (accuracy, loss, confusion matrix, and ROC curve multiclass)
    @Utilities.timer_func
    def figure_plot_four_multiclass(self) -> None: 
        """_summary_
        """

        # * Colors for ROC curves
        Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

        # *
        Four_plot = 'Four_plot'
        Roc_auc = dict()


        # * Figure's size
        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
        plt.suptitle(self._Title, fontsize = self._Font_size_title)
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 4)

        # * Confusion matrix heatmap
        sns.set(font_scale = self._Font_size_general)

        # *
        ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_confusion})
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        # * Subplot training accuracy
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)
        plt.plot(self._Accuracy, label = 'Training Accuracy')
        plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc = 'lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')

        # * Subplot training loss
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
        plt.plot(self._Loss, label = 'Training Loss')
        plt.plot(self._Val_loss, label = 'Validation Loss')
        plt.ylim([0, 2.0])
        plt.legend(loc = 'upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')

        # * FPR and TPR values for the ROC curve
        for i in range(len(self._Roc_curve_dataframes)):
            Roc_auc[i] = auc(self._FPRs[i], self._TPRs[i])

        # * Plot ROC curve
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 3)
        plt.plot([0, 1], [0, 1], 'k--')

        for i in range(len(self._Roc_curve_dataframes)):
            plt.plot(self._FPRs[i], self._TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self._Labels[i], Roc_auc[i]))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc = 'lower right')
        
        self.save_figure(self._Save_figure, self._Title, Four_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot two images in a figure (confusion matrix, and ROC curve). Special for machine learning algorithms.
    @Utilities.timer_func
    def figure_plot_two(self) -> None: 
        """_summary_
        """

        # *
        Two_plot = 'Two_plot'

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
        plt.suptitle(self._Title, fontsize = self._Font_size_title)
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)

        # * Confusion matrix heatmap
        sns.set(font_scale = self._Font_size_general)

        # *
        ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_confusion})
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        # * FPR and TPR values for the ROC curve
        Auc = auc(self._FPRs[0], self._TPRs[0])

        # * Subplot ROC curve
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self._FPRs[0], self._TPRs[0], label = 'Test' + '(area = {:.4f})'.format(Auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')
        
        self.save_figure(self._Save_figure, self._Title, Two_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot two images in a figure (confusion matrix, and ROC curve multiclass). Special for machine learning algorithms.
    @Utilities.timer_func
    def figure_plot_two_multiclass(self) -> None: 
        """_summary_
        """

        # * Colors for ROC curves
        Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

        # *
        Two_plot = 'Two_plot'

        Roc_auc = dict()

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
        plt.suptitle(self._Title, fontsize = self._Font_size_title)
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)

        # * Confusion matrix heatmap
        sns.set(font_scale = self._Font_size_general)

        # *
        ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_confusion})
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        # * FPR and TPR values for the ROC curve
        for i in range(len(self._Roc_curve_dataframes)):
            Roc_auc[i] = auc(self._FPRs[i], self._TPRs[i])

        # * Plot ROC curve
        plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
        plt.plot([0, 1], [0, 1], 'k--')

        for i in range(len(self._Roc_curve_dataframes)):
            plt.plot(self._FPRs[i], self._TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self._Labels[i], Roc_auc[i]))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc = 'lower right')
        
        self.save_figure(self._Save_figure, self._Title, Two_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot confusion matrix.
    @Utilities.timer_func
    def figure_plot_CM(self) -> None:
        """_summary_
        """
        
        # *
        CM_plot = 'CM_plot'

        # *
        Confusion_matrix_dataframe = pd.read_csv(self._CM_dataframe)

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
        plt.title('Confusion Matrix with {}'.format(self._Title))

        # * Confusion matrix heatmap
        sns.set(font_scale = self._Font_size_general)

        # *
        ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_confusion})
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        self.save_figure(self._Save_figure, self._Title, CM_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot accuracy.
    @Utilities.timer_func
    def figure_plot_acc(self) -> None:
        """_summary_
        """
        
        # *
        ACC_plot = 'ACC_plot'

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
        plt.title('Training and Validation Accuracy with {}'.format(self._Title))

        # * Plot training accuracy
        plt.plot(self._Accuracy, label = 'Training Accuracy')
        plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc = 'lower right')
        plt.xlabel('Epoch')

        self.save_figure(self._Save_figure, self._Title, ACC_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to plot loss.
    @Utilities.timer_func
    def figure_plot_loss(self) -> None:
        """_summary_
        """

        # *
        Loss_plot = 'Loss_plot'

        # * Figure's size
        
        plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
        plt.title('Training and Validation Loss with {}'.format(self._Title))

        # * Plot training loss
        plt.plot(self._Loss, label = 'Training Loss')
        plt.plot(self._Val_loss, label = 'Validation Loss')
        plt.ylim([0, 2.0])
        plt.legend(loc = 'upper right')
        plt.xlabel('Epoch')

        self.save_figure(self._Save_figure, self._Title, Loss_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to ROC curve.
    @Utilities.timer_func
    def figure_plot_ROC_curve(self) -> None:
        """_summary_
        """

        # *
        ROC_plot = 'ROC_plot'

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
        plt.title('ROC curve Loss with {}'.format(self._Title))

        # * FPR and TPR values for the ROC curve
        AUC = auc(self._FPRs[0], self._TPRs[0])

        # * Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self._FPRs[0], self._TPRs[0], label = 'Test' + '(area = {:.4f})'.format(AUC))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc = 'lower right')

        self.save_figure(self._Save_figure, self._Title, ROC_plot, self._Folder_path)
        self.show_figure(self._Show_image)

    # ? Method to ROC curve multiclass.
    @Utilities.timer_func
    def figure_plot_ROC_curve_multiclass(self) -> None:
        """_summary_
        """
        # * Colors for ROC curves
        Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

        # *
        ROC_plot = 'ROC_plot'
        Roc_auc = dict()

        # * Figure's size
        plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
        plt.title(self._Title, fontsize = self._Font_size_title)

        # * FPR and TPR values for the ROC curve
        for i in range(len(self._Roc_curve_dataframes)):
            Roc_auc[i] = auc(self._FPRs[i], self._TPRs[i])

        # * Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')

        for i in range(len(self._Roc_curve_dataframes)):
            plt.plot(self._FPRs[i], self._TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self._Labels[i], Roc_auc[i]))

            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc = 'lower right')

        self.save_figure(self._Save_figure, self._Title, ROC_plot, self._Folder_path)
        self.show_figure(self._Show_image)