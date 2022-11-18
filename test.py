import os

Files = []
Folders = []
Root_list = []
Folders_list = []

for _, Dirnames, Filenames in os.walk("D:\Mini-MIAS\CBIS_DDSM_NO_Images_Biclass"):
    Folders.append(_)

print(Folders)