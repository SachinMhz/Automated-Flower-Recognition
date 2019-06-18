
# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import glob
from os import path
import random,time
from PIL import ImageTk, Image
import numpy as np
import mahotas
import cv2
import os
import h5py
import matplotlib.pyplot as plt

# fixed-sizes for image
fixed_size = tuple((500, 500))

# path to training data
train_path = "dataset/train/"
#image_path
#mainMenuImg = "assets/interface_bg2.jpg"
uploadImg= "assets/upload.gif"
identifyImg = "assets/identify.gif"
splashScreenImg = "assets/bg.jpg"
mainMenuImg = "assets/interface_bg.jpg"

DaisyPath = "dataset/train/Daisy/"
DandelionPath = "dataset/train/Dandelion/"
RosePath = "dataset/train/Rose/"
SunflowerPath = "dataset/train/Sunflower/"
TulipPath = "dataset/train/Tulip/"

#checking for image Uploaded
test_path = 0
# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.30

# seed for reproducing same results
seed = 9

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    #convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #calculates Hu’s seven invariant moments.
    #Finally, we flatten our array to form our shape feature vector.
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# num of images per class
images_per_class =  100  #initally 80
