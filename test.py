import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r"D:\CBIS-DDSM\CBIS-DDSM Final\CBIS_DDSM_CLAHE_Images_Biclass\Calc_CLAHE_Abnormal_Images\Calc-Test_P_00038_LEFT_MLO_1_Benign_CLAHE.png",0)
  
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()