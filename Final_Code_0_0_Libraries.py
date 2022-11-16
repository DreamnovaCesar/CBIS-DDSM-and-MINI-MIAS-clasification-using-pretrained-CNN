# *
import os
import sys
import random
import datetime

# *
import numpy as np
from numpy import ndarray

# *
import pandas as pd
import seaborn as sns

# *
import tensorflow as tf
import tensorflow as keras
import matplotlib.pyplot as plt

# *
import random
from random import randint
from skimage import io

# *
from itertools import cycle

# *
import time

# *
import cv2
import string
import shutil
import pydicom
import warnings

# *
import splitfolders

# *
from sklearn.cluster import KMeans

# *
from random import sample

# *
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# *
from cryptography.fernet import Fernet

# *
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

# *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import normalized_mutual_information as nmi

# *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

# *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# *
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder

# *
from functools import wraps

# *
from keras import Input
from keras import datasets
from keras import layers
from keras import models

# *
from keras.optimizers import Adam

# *
from keras.applications import ResNet50
from keras.applications import ResNet50V2
from keras.applications import ResNet152
from keras.applications import ResNet152V2

# *
from keras.applications import MobileNet

# *
from keras.applications import MobileNetV3Small
from keras.applications import MobileNetV3Large

# *
from keras.applications import Xception

# *
from keras.applications import VGG16
from keras.applications import VGG19

# *
from keras.applications import InceptionV3
from keras.applications import DenseNet121
from keras.applications import DenseNet201

# *
from keras.applications import Xception
from keras.applications import NASNetLarge

# *
from keras.applications import EfficientNetB0
from keras.applications import EfficientNetB1
from keras.applications import EfficientNetB2
from keras.applications import EfficientNetB3
from keras.applications import EfficientNetB4
from keras.applications import EfficientNetB5
from keras.applications import EfficientNetB6
from keras.applications import EfficientNetB7

# *
from keras.models import Sequential
from keras.models import Model

# *
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU

# *
from keras import regularizers

# *
from sklearn.model_selection import train_test_split

# *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau

# *
from keras.preprocessing.image import ImageDataGenerator

# *
from skimage import io
from skimage import filters
from skimage import img_as_ubyte
from skimage import img_as_float

# *
from skimage.exposure import equalize_adapthist
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity

# *
from skimage.filters import unsharp_mask

# *
import albumentations as A

# *
from glrlm import GLRLM
from skimage.feature import graycomatrix, graycoprops

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier

#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

from tensorflow.python.client import device_lib
