import tkinter
import tkinter.messagebox
import customtkinter

from tkinter import filedialog

from Final_Code_0_0_Libraries import *
from Final_Code_0_1_Class_Utilities import Utilities

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

#https://github.com/TomSchimansky/CustomTkinter

# ?
class MenuTkinter(Utilities):

    """
    """

    @Utilities.time_func  
    def menu(self):

        app = App();
        app.mainloop();

# ?
class App(customtkinter.CTk):

    WIDTH = 780;
    HEIGHT = 700;

    def __init__(self):
        super().__init__()

        # * set default values padding
        self.__Padding_button_x = 20;
        self.__Padding_button_y = 20;

        self.title("Article Euler 2D and 3D");
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}");
        self.protocol("WM_DELETE_WINDOW", self.on_closing);  # call .on_closing() when app gets closed

        # ================================================ create two frames ================================================

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight = 1);
        self.grid_rowconfigure(0, weight = 1);

        self.frame_left = customtkinter.CTkFrame(master = self,
                                                 width = 180,
                                                 corner_radius = 0);
        self.frame_left.grid(row = 0, column = 0, sticky = "nswe");

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        # ================================================ frame_left ================================================

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize = 10);   # empty row with minsize as spacing
        #self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize = 20);    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize = 10);  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master = self.frame_left,
                                              text = "Options",
                                              text_font = ("Roboto Medium", -16));  # font name and size in px
        self.label_1.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 button left frame
        self.button_C2DI = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "Create 2D images",
                                                    command = self.button_change_menu_create_objects_2D);
        self.button_C2DI.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 button left frame
        self.button_C3DI = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "Create 3D images",
                                                    command = self.button_change_menu_create_objects_3D);
        self.button_C3DI.grid(row = 3, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 3 button left frame
        self.button_T2DM = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "Training 2D model",
                                                    command = self.button_change_menu_train_models_2D);
        self.button_T2DM.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 4 button left frame
        self.button_T3DM = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "Training 3D model",
                                                    command = self.button_change_menu_train_models_3D);
        self.button_T3DM.grid(row = 5, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 5 button left frame
        self.button_P2D = customtkinter.CTkButton(  master = self.frame_left,
                                                    text = "Prediction 2D",
                                                    command = self.button_change_menu_prediction_2D);
        self.button_P2D.grid(row = 6, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 6 button left frame
        self.button_P3D = customtkinter.CTkButton(  master = self.frame_left,
                                                    text = "Prediction 3D",
                                                    command = self.button_change_menu_prediction_3D);
        self.button_P3D.grid(row = 7, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 7 button left frame
        self.button_BACK = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "BACK",
                                                    command = self.button_back_menu);
        self.button_BACK.grid(row = 8, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 8 button appearance mode: (Black and white)
        self.label_mode = customtkinter.CTkLabel(   master = self.frame_left, 
                                                    text = "Appearance Mode:");
        self.label_mode.grid(row = 9, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "w");

        # * 8 button appearance mode: (Black and white) options
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master = self.frame_left,
                                                        values = ["Light", "Dark", "System"],
                                                        command = self.change_appearance_mode);
        self.optionmenu_1.grid(row = 10, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "w");

    # * set default values
        self.optionmenu_1.set("Dark")
    
    # ?
    def button_change_menu_create_objects_2D(self):

        # =============================================== frame_right ===============================================
        
        # * Destroy frame_right and create another one
        self.frame_right.destroy();

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        self.frame_info.rowconfigure(0, weight = 2);
        self.frame_info.columnconfigure(1, weight = 2);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Create 2D images",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Entry(Add number of objects)
        self.Entry_NO = customtkinter.CTkEntry( self.Labelframe_data, 
                                                width = 150,
                                                placeholder_text = "Number of objects");
        self.Entry_NO.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Entry(Add height of each object)
        self.Entry_height = customtkinter.CTkEntry( self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Height");
        self.Entry_height.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);
        
        # * 3 Entry(Add width of each object)
        self.Entry_width = customtkinter.CTkEntry(  self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Width");
        self.Entry_width.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Button to recolect the information from entries.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.create_objects_2D);

        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_train_models_2D(self):

        # =============================================== frame_right ===============================================
        
        # * Destroy frame_right and create another one
        self.frame_right.destroy();

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        self.frame_info.rowconfigure(0, weight = 2);
        self.frame_info.columnconfigure(1, weight = 2);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Train 2D model",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Entry(Model name)
        self.Entry_model_name = customtkinter.CTkEntry( self.Labelframe_data, 
                                                width = 150,
                                                placeholder_text = "Model name");
        self.Entry_model_name.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Entry(Epochs)
        self.Entry_epochs = customtkinter.CTkEntry( self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Epochs");
        self.Entry_epochs.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Combobox()
        self.Combobox_model = customtkinter.CTkComboBox(master = self.Labelframe_data,
                                                        values = ["Connectity 4", "Connectity 8"])
        self.Combobox_model.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x)

        # * 1 Button to recolect the information from entries.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.train_model_2D);

        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_prediction_2D(self):

        # =============================================== frame_right ===============================================
        
        # * Destroy frame_right and create another one
        self.frame_right.destroy();

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        self.frame_info.rowconfigure(0, weight = 2);
        self.frame_info.columnconfigure(1, weight = 2);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Prediction using 2D model",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Button to load .h5
        self.Button_model = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Choose model!",
                                                        command = self.open_txt);
        self.Button_model.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Button to load .txt
        self.Button_txt = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Choose .txt!",
                                                        command = self.open_txt);
        self.Button_txt.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 3 Button to recolect the information.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.prediction_2D);
        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_create_objects_3D(self):

        # =============================================== frame_right ===============================================

        self.frame_right.destroy();

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight = 1);
        self.frame_info.columnconfigure(1, weight = 1);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Create 3D images",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Entry(Add number of objects)
        self.Entry_NO = customtkinter.CTkEntry( self.Labelframe_data, 
                                                width = 150,
                                                placeholder_text = "Number of objects");
        self.Entry_NO.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Entry(Add height of each object)
        self.Entry_height = customtkinter.CTkEntry( self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Height");
        self.Entry_height.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);
        
        # * 3 Entry(Add width of each object)
        self.Entry_width = customtkinter.CTkEntry(  self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Width");
        self.Entry_width.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 4 Entry(Add depth of each object)
        self.Entry_depth = customtkinter.CTkEntry(  self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Depth");
        self.Entry_depth.grid(row = 3, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Button to recolect the information from entries.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.create_objects_3D);

        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_train_models_3D(self):

        # =============================================== frame_right ===============================================
        
        # * Destroy frame_right and create another one
        self.frame_right.destroy();

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        self.frame_info.rowconfigure(0, weight = 2);
        self.frame_info.columnconfigure(1, weight = 2);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");
        
        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Train 3D model",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Entry(Model name)
        self.Entry_model_name = customtkinter.CTkEntry( self.Labelframe_data, 
                                                width = 150,
                                                placeholder_text = "Model name");
        self.Entry_model_name.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Entry(Epochs)
        self.Entry_epochs = customtkinter.CTkEntry( self.Labelframe_data, 
                                                    width = 150,
                                                    placeholder_text = "Epochs");
        self.Entry_epochs.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Combobox()
        self.Combobox_model = customtkinter.CTkComboBox(master = self.Labelframe_data,
                                                        values = ["MLP", "RF"])
        self.Combobox_model.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x)

        if(self.Combobox_model.get() == 'RF'):

            self.Entry_epochs.configure(state = "disabled", text = "Disabled")

        # * 1 Button to recolect the information from entries.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.train_model_3D);

        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_prediction_3D(self):

        # =============================================== frame_right ===============================================
        
        # * Destroy frame_right and create another one
        self.frame_right.destroy();
        
        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

        self.frame_info = customtkinter.CTkFrame(master = self.frame_right);
        self.frame_info.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nsew");

        # =============================================== frame_info ===============================================

        self.frame_info.rowconfigure(0, weight = 2);
        self.frame_info.columnconfigure(1, weight = 2);

        self.Labelframe_data = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        self.Labelframe_info = customtkinter.CTkFrame(self.frame_info);
        self.Labelframe_info.grid(row = 1, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky="nsew");

        # * Label
        self.Label_stage = customtkinter.CTkLabel(  master = self.frame_info,
                                                    text = "Prediction using 3D model",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Button to load .h5
        self.Button_model = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Choose model!",
                                                        command = self.open_txt);
        self.Button_model.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 Button to load .txt
        self.Button_txt = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Choose .txt!",
                                                        command = self.open_txt);
        self.Button_txt.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 3 Button to recolect the information.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.prediction_3D);
        self.Button_add_data.grid(row = 4, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_back_menu(self):

        self.frame_right = customtkinter.CTkFrame(master = self);
        self.frame_right.grid(row = 0, column = 1, pady = self.__Padding_button_y, padx = self.__Padding_button_x, sticky = "nswe");

    # ?
    def button_NU(self):
        print('Empty function')

    # ?
    def create_objects_2D(self):
        Images_2D = DataEuler(folder = self.__Euler_path_images_2D, NI = self.Entry_NO.get(), 
                                Height = self.Entry_height.get(), Width = self.Entry_width.get());

        Images_2D.create_data_euler_2D_random();

    # ?
    def train_model_2D(self):

        if(self.Combobox_model.get() == 'Connectity 4'):

            Euler_2D = EulerNumberML2D( input = Input_2D, output = Output_2D_4_Connectivity, 
                                            folder = self.__Euler_path_data_2D, MN = self.Entry_model_name.get(), epochs = self.Entry_epochs.get());

            Euler_2D.model_euler_MLP_2D();
        
        elif(self.Combobox_model.get() == 'Connectity 8'):

            Euler_2D = EulerNumberML2D( input = Input_2D, output = Output_2D_8_Connectivity, 
                                            folder = self.__Euler_path_data_2D, MN = self.Entry_model_name.get(), epochs = self.Entry_epochs.get());

            Euler_2D.model_euler_MLP_2D();

    # ?
    def prediction_2D(self):

        Euler_2D = EulerNumberML2D()

        Array = Euler_2D.obtain_arrays_from_object_2D(self.Button_txt.get())
        Euler_2D.model_prediction_2D(self.Button_model.get(), Array)

    # ?
    def create_objects_3D(self):
        Images_3D = DataEuler(folder = self.__Euler_path_images_3D, NI = self.Entry_NO.get(), 
                                Height = self.Entry_height.get(), Width = self.Entry_width.get(), Depth = self.Entry_depth.get());

        Images_3D.create_data_euler_3D_random();

    # ?
    def train_model_3D(self):

        if(self.Combobox_model.get() == 'MLP'):

            Euler_3D = EulerNumberML3D( input = Input_3D_array, output = Output_3D_array, 
                                        folder = self.__Euler_path_data_3D, MN = self.Entry_model_name.get(), epochs = self.Entry_epochs.get());

            Euler_3D.model_euler_MLP_3D();
        
        elif(self.Combobox_model.get() == 'RF'):

            Euler_3D = EulerNumberML3D( input = Input_3D_array, output = Output_3D_array, 
                                        folder = self.__Euler_path_data_3D, MN = self.Entry_model_name.get(), epochs = self.Entry_epochs.get());

            Euler_3D.model_euler_RF_3D();

    # ?
    def prediction_3D(self):

        Euler_3D = EulerNumberML3D()

        Array = Euler_3D.obtain_arrays_from_object_3D(self.Button_txt.get())
        Euler_3D.model_prediction_3D(self.Button_model.get(), Array)

    # ?
    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def open_txt(event = None):
        Filename = filedialog.askopenfilename()
        print('Selected:', Filename)

        
    # ?
    def on_closing(self, event = 0):
        self.destroy()
