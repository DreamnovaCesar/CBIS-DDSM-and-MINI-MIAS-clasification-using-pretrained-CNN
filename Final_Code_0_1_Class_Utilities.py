
from Final_Code_0_0_Libraries import time
from Final_Code_0_0_Libraries import tf
from Final_Code_0_0_Libraries import wraps

# ? Utilities

class Utilities(object):
    """
    Class used to store decorators.

    Methods:
        timer_func(func): This function saves the time it takes for the function to finish its process (Decorator).

        detect_GPU(func): This function analyzes if there is a gpu in the system for its use (Decorator).
    """

    # ? Get the execution time of each function
    @staticmethod  
    def timer_func(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * Obtain the executed time of the function

            Asterisk = 60;

            t1 = time.time();
            result = func(self, *args, **kwargs);
            t2 = time.time();

            print("\n");
            print("*" * Asterisk);
            print('Function {} executed in {:.4f}'.format(func.__name__, t2 - t1));
            print("*" * Asterisk);
            print("\n");

            return result
        return wrapper
    
    # ? Detect fi GPU exist in your PC for CNN Decorator
    @staticmethod  
    def detect_GPU(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * Obtain the executed time of the function
            GPU_name = tf.test.gpu_device_name();
            GPU_available = tf.config.list_physical_devices();
            print("\n");
            print(GPU_available);
            print("\n");
            #if GPU_available == True:
                #print("GPU device is available")
            if "GPU" not in GPU_name:
                print("GPU device not found");
                print("\n");
            print('Found GPU at: {}'.format(GPU_name));
            print("\n");

            result = func(self, *args, **kwargs);

            return result
        return wrapper