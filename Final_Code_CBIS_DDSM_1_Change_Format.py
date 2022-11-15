
from Final_Code_1_General_Functions import DCM_format

def preprocessing_ChangeFormat() -> None:

    Calcification = 'Calficication'
    Mass = 'Mass'

    Test = 'Test'
    Training = 'Training'


    # * With this class we change the format of each image for a new one
    DCM_Calc_Test = DCM_format(Folder = '', Datafolder = '', Normalfolder = '', 
                                Resizefolder = '', Normalizefolder = '', 
                                    Severity = Calcification, Phase = Test)

    DCM_Calc_Training = DCM_format(Folder = '', Datafolder = '', Normalfolder = '', 
                                Resizefolder = '', Normalizefolder = '',
                                        Severity = Calcification, Phase = Training)

    DCM_Mass_Test = DCM_format(Folder = '', Datafolder = '', Normalfolder = '', 
                                Resizefolder = '', Normalizefolder = '',    
                                    Severity = Mass, Phase = Test)

    DCM_Mass_Training = DCM_format(Folder = '', Datafolder = '', Normalfolder = '', 
                                Resizefolder = '', Normalizefolder = '',
                                    Severity = Mass, Phase = Training)


    DCM_Calc_Test.DCM_change_format()
    DCM_Calc_Training.DCM_change_format()
    DCM_Mass_Test.DCM_change_format()
    DCM_Mass_Training.DCM_change_format()