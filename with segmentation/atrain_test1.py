
# import the necessary packages
#from global_test import *
from areq_dec1 import *
import glob
from PIL import ImageTk, Image
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
import numpy as np

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print ("[STATUS] features shape: {}".format(global_features.shape))
print ("[STATUS] labels shape: {}".format(global_labels.shape))
print ("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from os import path
import random,time
from PIL import ImageTk, Image

def SplashScreen():
    root = Tk()
    root.overrideredirect(True)
    root.geometry('1024x720')
    path = "assets/bg3.jpg"
    img = ImageTk.PhotoImage(Image.open(path))
    panel = Label(root, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    root.after(1000, root.destroy)
    root.mainloop()

#SplashScreen()

window=Tk()
window.title("Automated Flower Recognition")
window.geometry('1024x720')
interface_bg_path = "assets/interface_bg.jpg"
img = ImageTk.PhotoImage(Image.open(interface_bg_path))
panel = Label(window,image=img)
panel.pack(side = "bottom",fill ="both", expand ="yes")
#window.configure(background='#58FA82')
#HeadingLabel = Label(window, text="Automated Flower Recognition", font=("Arial Bold", 50))
#HeadingLabel.pack()

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=9)
clf2 = svm.SVC(kernel='rbf', C=1,gamma=10)
clf3 = LogisticRegression(random_state=9)
clf4 = LinearDiscriminantAnalysis()
clf5 = KNeighborsClassifier()
clf6 = DecisionTreeClassifier(random_state=9)
clf7 = GaussianNB()
#clf7 = SVC(random_state=9)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)
clf2.fit(trainDataGlobal, trainLabelsGlobal)
clf3.fit(trainDataGlobal, trainLabelsGlobal)
clf4.fit(trainDataGlobal, trainLabelsGlobal)
clf5.fit(trainDataGlobal, trainLabelsGlobal)
clf6.fit(trainDataGlobal, trainLabelsGlobal)
#clf7.fit(trainDataGlobal, trainLabelsGlobal)

def UploadClicked():
    global test_path
    test_path = filedialog.askopenfilename()
    ShowText(window,test_path,650,370,0,0,8)
    #show selected image in the window
    img = Image.open((test_path))
    img = img.resize((350, 260), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img
    panel.pack()
    panel.place(x=650,y=80)

def ShowDetails(flower_predicted, prediction_array, test_path):
    showDetails = Toplevel(window)
    #flower_predicted = "Dandelion"
    Algorithm_name = ["RandomForestClassifier \n","SVM with parameter \n","LogisticRegression \n","KNeighborsClassifier \n","DecisionTreeClassifier \n","GaussianNB \n"]
    showDetails.title("About the Flower:-  " + flower_predicted )
    showDetails.geometry('1024x720')
    #showDetails.configure(background='grey')
    path = 'dataset/train/'+flower_predicted+"/"+flower_predicted+".jpg"
    # Background images
    ShowImage(showDetails,path,0,0,1024,720)
    # Flower image
    ShowImage(showDetails,test_path,600,20,350,260)
    # show test in the show deatails window
    #ShowText(showDetails,Algorithm_name,200,10,0,0,14)
    #ShowText(showDetails,flower_predicted,100,500,0,0,20)
    ShowText(showDetails,prediction_array,200,10,0,0,20)
    showDetails.mainloop()

def ShowText(window,Text,X,Y,Height, Width, FontSize, Font="Arial Bold"):
    textLabel = Label(window, text=Text, font = (Font,FontSize), height=Height,width=Width)
    textLabel.pack()
    textLabel.place(x=X, y=Y)
def ShowImage(window,Path,X,Y,Height,Width):
    img = Image.open((Path))
    img = img.resize((Height, Width), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img
    panel.pack()
    panel.place(x=X,y=Y)
def ShowButton(Window,Text,Command,X,Y,Height,Width):
    Btn = Button(Window, text=Text, command=Command, height=Height, width=Width)
    Btn.pack()
    Btn.place(x=X, y=Y)

def most_common(lst):
    return max(set(lst), key=lst.count)

def IdentifyClicked():
    # read the image
    image = cv2.imread(test_path)
    # resize the image
    image = cv2.resize(image, fixed_size)

    mask = np.zeros(image.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (10,10,490,450)
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]


    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    prediction2 = clf2.predict(global_feature.reshape(1,-1))[0]
    prediction3 = clf3.predict(global_feature.reshape(1,-1))[0]
    prediction4 = clf4.predict(global_feature.reshape(1,-1))[0]
    prediction5 = clf5.predict(global_feature.reshape(1,-1))[0]
    prediction6 = clf6.predict(global_feature.reshape(1,-1))[0]
    #prediction7 = clf7.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    #cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    #array for predictions
    prediction_array=[]
    prediction_array.append(train_labels[prediction])
    prediction_array.append(train_labels[prediction2])
    prediction_array.append(train_labels[prediction3])
    prediction_array.append(train_labels[prediction4])
    prediction_array.append(train_labels[prediction5])
    prediction_array.append(train_labels[prediction6])
    #prediction_array.append(train_labels[prediction7])

    #flower_predicted = most_common(prediction_array)
    flower_predicted = train_labels[prediction2]
    # display the output image
    ShowDetails(flower_predicted,prediction_array, test_path)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()

#show text in UI
#ShowText(window,"Automated Flower Recognition",175, 0, 0, 0, 35)
#ShowText(window,"Select the flower you \n want to identify ", 650, 80, 8, 20, 20)
#ShowText(window,"OUR APPLICATION \n INFORMATION HERE....",50,80,9,20,35)

#make BUtton on the UI
ShowButton(window,"Upload", UploadClicked,775,400,4,15)
ShowButton(window,"Identify",IdentifyClicked,775,500,4,15)

window.mainloop()
