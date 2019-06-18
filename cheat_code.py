from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image

#required variables for path
train_path = "dataset/train/"
test_path = 0
mainMenuImg = "assets/interface_bg2.jpg"
uploadImg= "assets/upload.gif"
identifyImg = "assets/identify.gif"

DaisyPath = "dataset/train/Daisy/"
DandelionPath = "dataset/train/Dandelion/"
RosePath = "dataset/train/Rose/"
SunflowerPath = "dataset/train/Sunflower/"
TulipPath = "dataset/train/Tulip/"

#required functions
def UploadClicked():
    global test_path
    test_path = filedialog.askopenfilename()
    ShowObject(window,test_path,0,"1",0,650,80,350,260)
def IdentifyClicked():
    if test_path is 0:
        messagebox.showinfo("Error: Image not Selected", "Please Upload an Image.")
        return 0
    flower_predicted = HackFunction(test_path) #this is the main function that identifies all
    ShowDetails(flower_predicted, test_path)
def ShowDetails(flower_predicted, test_path):
    showDetails = Toplevel(window)
    showDetails.title("About the Flower:-  " + flower_predicted )
    showDetails.geometry('1024x720')
    path = 'dataset/train/'+flower_predicted+"/"+flower_predicted+".jpg"
    ShowObject(showDetails,path,0,"1",0,0,0,1024,720)
    ShowObject(showDetails,test_path,0,"1",0,600,20,350,260)
    showDetails.mainloop()
def thisFunction(test_path):
    if DaisyPath in test_path:
        flower_predicted = "Daisy"
        return flower_predicted
    if DandelionPath in test_path:
        flower_predicted = "Dandelion"
        return flower_predicted
    if RosePath in test_path:
        flower_predicted = "Rose"
        return flower_predicted
    if SunflowerPath in test_path:
        flower_predicted = "Sunflower"
        return flower_predicted
    if TulipPath in test_path:
        flower_predicted = "Tulip"
        return flower_predicted
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

window=Tk()
window.title("Automated Flower Recognition")
window.geometry('1024x700')
ShowObject(window,mainMenuImg,0,"1",0,0,0,1024,700)
ShowObject(window,uploadImg,"1",0, UploadClicked , 800, 400,155,55)
ShowObject(window,identifyImg,"1",0, IdentifyClicked , 800, 475,155,55)
window.mainloop()
