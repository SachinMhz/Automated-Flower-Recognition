
# import the necessary packages
from req_decl import *
#function declatation
def SplashScreen():
    root = Tk()
    root.overrideredirect(True)
    root.geometry('1024x700')
    ShowObject(root,splashScreenImg,0,"1",0,0,0,1024,700)
    root.after(1000, root.destroy)
    root.mainloop()
def UploadClicked():
    global test_path
    test_path = filedialog.askopenfilename()
    ShowText(window,test_path,550,370,0,0,8)
    ShowObject(window,test_path,0,"1",0,650,80,350,260)
def IdentifyClicked():
    # read the image
    if test_path is 0:
        messagebox.showinfo("Error: Image not Selected", "Please Upload an Image.")
        return 0

    window2 = Tk()
    window2.title("Algorithm's Prediction")
    window2.geometry("300x300")
    image = cv2.imread(test_path)
    image = cv2.resize(image, fixed_size)

    # Global Feature extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    # Concatenate global features
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    prediction2 = clf2.predict(global_feature.reshape(1,-1))[0]
    #prediction3 = clf3.predict(global_feature.reshape(1,-1))[0]
    #prediction4 = clf4.predict(global_feature.reshape(1,-1))[0]
    prediction5 = clf5.predict(global_feature.reshape(1,-1))[0]
    prediction6 = clf6.predict(global_feature.reshape(1,-1))[0]
    #prediction7 = clf7.predict(global_feature.reshape(1,-1))[0]

    #array for predictions
    prediction_array=[]
    prediction_array.append(train_labels[prediction])
    prediction_array.append(train_labels[prediction2])
    #prediction_array.append(train_labels[prediction3])
    #prediction_array.append(train_labels[prediction4])
    prediction_array.append(train_labels[prediction5])
    prediction_array.append(train_labels[prediction6])
    #prediction_array.append(train_labels[prediction7])

    Algorithm_name = ["RandomForestClassifier \n","SVM with parameter \n",
                        "KNeighborsClassifier \n","GaussianNB \n"]
    ShowText(window2, Algorithm_name, 0,0,0,0,8)
    ShowText(window2, prediction_array,0,100,0,0,8)
    flower_predicted = most_common(prediction_array)
    #flower_predicted = SckitLearnProcess(test_path,flower_predicted)
    # display the output image
    ShowDetails(flower_predicted,prediction_array, test_path)
def ShowDetails(flower_predicted, prediction_array, test_path):
    showDetails = Toplevel(window)
    showDetails.title("About the Flower:-  " + flower_predicted )
    showDetails.geometry('1024x720')
    #flower_predicted = "Rose"
    print(flower_predicted)
    path = 'dataset/train/'+flower_predicted+"/"+flower_predicted+".jpg"
    # Background images
    ShowObject(showDetails,path,0,"1",0,0,0,1024,720)
    ShowObject(showDetails,test_path,0,"1",0,600,20,350,260)
    # show test in the show deatails window
    ShowText(showDetails,Algorithm_name,200,50,0,0,14)
    #ShowText(showDetails,flower_predicted,100,500,0,0,20)
    ShowText(showDetails,prediction_array,200,10,0,0,20)
    showDetails.mainloop()
def ShowAccuracy():
    #create all the machine learning models
    models = []
    #models.append(('LR', LogisticRegression(random_state=9)))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=9)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', svm.SVC(kernel='rbf', C=1,gamma=10)))
    #models.append(('SVM', SVC(random_state=9)))
    # variables to hold the results and names
    results = []
    names = []
    scoring = "accuracy"
    model_accuracy = []
    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')
    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        model_accuracy.append(cv_results.mean())
    model_accuracy = str(max(model_accuracy)*100)
    ShowText(window,"Our Model is "+ model_accuracy +"% accurate using current dataset training.",500,675,0,0,8 )
def most_common(list):
    return max(set(list), key=list.count)
def ShowText(window,Text,X,Y,Height, Width, FontSize, Font="Arial Bold"):
    textLabel = Label(window, text=Text, font = (Font,FontSize), height=Height,width=Width)
    textLabel.pack()
    textLabel.place(x=X, y=Y)
def ShowObject(window, Path, button, image ,Command ,X , Y , Height , Width):
    img = Image.open((Path))
    img = img.resize((Height, Width), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    if image is "1":
        panel = Label(window, image=img)
    if button is "1":
        panel = tk.Button(window, width=155, height=55, image=img,command=Command,highlightthickness = 0,borderwidth=-1)
    panel.image = img
    panel.pack()
    panel.place(x=X,y=Y)

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

SplashScreen()
window=Tk()
window.title("Automated Flower Recognition")
window.geometry('1024x700')

# create the model - Random Forests and other Algorithms
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=9)
clf2 = svm.SVC(kernel='rbf', C=1,gamma=10)
clf5 = KNeighborsClassifier()
clf6 = DecisionTreeClassifier(random_state=9)
clf7 = GaussianNB()
#clf3 = LogisticRegression(random_state=9)
#clf4 = LinearDiscriminantAnalysis()
#clf7 = SVC(random_state=9)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)
clf2.fit(trainDataGlobal, trainLabelsGlobal)
clf5.fit(trainDataGlobal, trainLabelsGlobal)
clf6.fit(trainDataGlobal, trainLabelsGlobal)
#clf4.fit(trainDataGlobal, trainLabelsGlobal)
#clf3.fit(trainDataGlobal, trainLabelsGlobal)
#clf7.fit(trainDataGlobal, trainLabelsGlobal)

ShowObject(window,mainMenuImg,0,"1",0,0,0,1024,700)
ShowAccuracy()
ShowObject(window,uploadImg,"1",0, UploadClicked , 800, 400,155,55)
ShowObject(window,identifyImg,"1",0, IdentifyClicked , 800, 475,155,55)

window.mainloop()
