
from Final_Code_0_0_Libraries import *
from Final_Code_1_General_Functions_Classes import Utilities

class Menu(Utilities):
    """
    Utilities inheritance: Menu to show the functions for this research

    Methods:
        data_dic(): description
        
    """

    def __init__(self, **kwargs) -> None:
        pass

    # * Class variables
    def __repr__(self):
            return f'[]';

    # * Class description
    def __str__(self):
        return  f'';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Menu class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder path': str(),
                'Folder model': str(),
                };
 
 
    @Utilities.time_func  
    def menu(self):

        while(True):
            
            Asterisk = 60;

            print("\n");

            print("*" * Asterisk);
            print('What do you want to do:');
            print("*" * Asterisk);
            print('\n');
            print('1: ');
            print('2: ');
            print('\n');
            print("*" * Asterisk);
            print('\n');

            try: 
                Options = input('Option: ');

                if(Options == '1'):
                    pass

                elif(Options == '2'):
                    pass

                elif(Options == 'c'):

                    break;

            except:
                pass