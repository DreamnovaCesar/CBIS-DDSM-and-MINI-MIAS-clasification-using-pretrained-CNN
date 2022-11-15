
from Final_Code_0_0_Libraries import *
from Final_Code_1_General_Functions_Classes import Utilities

class Menu(Utilities):

    def __init__(self, **kwargs) -> None:
        pass

    def __repr__(self):

        kwargs_info = '';

        return kwargs_info

    def __str__(self):
        pass
 
 
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