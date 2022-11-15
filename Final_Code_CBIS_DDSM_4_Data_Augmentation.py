
from Final_Code_0_0_Libraries import *
from Final_Code_0_12_Data_Augmentation import DataAugmentation

# ? Data augmentation for CNN using RAM

# *
def Data_augmentation_stage(Folder_path: str, Labels: list[str], Numbers_iter: list[int]) -> None:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_path (str): _description_
        Labels (list[str]): _description_
        Numbers_iter (list[int]): _description_
    """
    # *
    Total_files = 0
    Total_dir = 0

    # *
    Total_images = 0
    Total_labels = 0

    # *
    Dir_total = []
    Folder_path_classes = []

    # *
    Object_DA = []

    # *
    for Base, Dirs, Files in os.walk(Folder_path):
        print('Searching in : ', Base)
        for Dir in Dirs:
            Dir_total.append(Dir)
            Total_dir += 1
        for Index, File in enumerate(Files):
            Total_files += 1

    # *
    for Index, dir in enumerate(Dir_total):
        Folder_path_classes.append('{}{}'.format(Folder_path, dir))
        print(Folder_path_classes[Index])

    for i in range(len(Folder_path_classes)):
        Object_DA.append(DataAugmentation(Folder = Folder_path_classes[i], NewFolder = Folder_path_classes[i], 
                                            Severity = Labels[i], Sampling = Numbers_iter[i], Label = i, SI = True))

    # *
    for i in range(len(Object_DA)):

        Images_, Labels_ = Object_DA[i].data_augmentation_same_folder() 

        Total_images += len(Images_)
        Total_labels += len(Labels_)

    print(len(Total_images))
    print(len(Total_labels))


def preprocessing_DataAugmentation_Folder(Folder_path: str, Labels: list[str], Numbers_iter: list[int], DA_T: bool = False, DA_V: bool = False) -> None:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_path (str): Path to save the new images
        Labels (list[str]): Class labels
        Numbers_iter (list[int]): Number of transformation per image
        DA_T (bool, optional): Data augmentation for test set. Defaults to False.
        DA_V (bool, optional): Data augmentation for validation set. Defaults to False.

    Raises:
        ValueError: _description_
    """
    # *
    Folder_path_train ='{}/train/'.format(Folder_path)
    Folder_path_val ='{}/val/'.format(Folder_path)
    Folder_path_test ='{}/test/'.format(Folder_path)

    # *
    if((len(Labels) != len(Numbers_iter))):
        raise ValueError("the length of one parameter is no equal") #! Alert

    # *
    Data_augmentation_stage(Folder_path_train, Labels, Numbers_iter)

    # *
    if(DA_V is True):

        Data_augmentation_stage(Folder_path_val, Labels, Numbers_iter)

    if(DA_T is True):

        Data_augmentation_stage(Folder_path_test, Labels, Numbers_iter)

