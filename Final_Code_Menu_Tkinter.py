import tkinter
import tkinter.messagebox
import customtkinter

from tkinter import filedialog

from Final_Code_0_0_Libraries import *
from Final_Code_0_0_Template_General_Functions_Classes import Utilities
from Final_Code_0_3_Class_Generation_Data import Generator
from Final_Code_0_5_Class_Change_Format import ChangeFormat

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

#https://github.com/TomSchimansky/CustomTkinter

# ?
class MenuTkinter(Utilities):

    """
    """

    @Utilities.timer_func  
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
                                                    text = "Create folders",
                                                    command = self.button_change_menu_create_folders);
        self.button_C2DI.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 2 button left frame
        self.button_CFI = customtkinter.CTkButton( master = self.frame_left,
                                                    text = "Change format",
                                                    command = self.button_change_menu_create_folders);
        self.button_CFI.grid(row = 2, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);


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
    def button_change_menu_create_folders(self):

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
                                                    text = "Create folders",
                                                    text_font = ("Roboto Medium", -16));  # font name and size in px
        self.Label_stage.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Entry(Add number of objects)
        self.Entry_CF = customtkinter.CTkEntry( self.Labelframe_data, 
                                                width = 150,
                                                placeholder_text = "Name of the folder");
        self.Entry_CF.grid(row = 0, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

        # * 1 Button to recolect the information from entries.
        self.Button_add_data = customtkinter.CTkButton( self.Labelframe_data,
                                                        text = "Go!",
                                                        command = self.create_folders_func);

        self.Button_add_data.grid(row = 1, column = 0, pady = self.__Padding_button_y, padx = self.__Padding_button_x);

    # ?
    def button_change_menu_change_format(self):

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
                                                        command = self.change_format_images);

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
    def create_folders_func(self):
        CF = Generator(folder = "D:\Test2", FN = self.Entry_CF.get())
        CF.create_folders()

    # ?
    def change_format_images_func(self):
        CFI = ChangeFormat(folder = '', newfolder = '', format = '', newformat = '',)

    # ?
    def change_DCM_format_func(self):
        pass
    
    # ?
    def change_DCM_format_func(self):
        pass
    
    # ?
    def crop_images_func(self):
        pass
    
    # ?
    def split_folders_func(self):
        pass
    
    # ?
    def figure_adjust_func(self):
        pass
    
    # ?
    def extract_features_func(self):
        pass
    
    # ?
    def image_processing_func(self):
        pass
    
    # ?
    def data_agumentation_func(self):
        pass

    # ?
    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def open_txt(event = None):
        Filename = filedialog.askopenfilename()
        print('Selected:', Filename)

        
    # ?
    def on_closing(self, event = 0):
        self.destroy()
