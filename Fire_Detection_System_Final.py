#importing all libraries
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading the datase
fire = glob.glob('1/*.jpg')
Nonfire = glob.glob('0/*.jpg')

#merging the two datasets
fire_list = []
nonfire_list = []
for x in fire:
  fire_list.append([x,"Fire"])
for x in Nonfire:
  nonfire_list.append([x,"No_fire"])

dataset = fire_list+nonfire_list 


#Creating a Dataframe
df = pd.DataFrame(dataset,columns = ['image','label'])
print(df.head(2))

#print(df.shape)

df = df.sample(frac=1).reset_index(drop=True)
df.head()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

Train_Generator = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.3,
                                    zoom_range=0.2,
                                    brightness_range=[0.2,0.9],
                                    rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.1)

Test_Generator = ImageDataGenerator(rescale=1./255)

from sklearn.model_selection import train_test_split
Train_Data,Test_Data = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)

#print("TRAIN SHAPE: ",Train_Data.shape)
#print("TEST SHAPE: ",Test_Data.shape)



Train_IMG_Set = Train_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="image",
                                                   y_col="label",
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="training")

Validation_IMG_Set = Train_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="image",
                                                   y_col="label",
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="validation")

Test_IMG_Set = Test_Generator.flow_from_dataframe(dataframe=Test_Data,
                                                 x_col="image",
                                                 y_col="label",
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)

for data_batch,label_batch in Train_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break

for data_batch,label_batch in Validation_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break

for data_batch,label_batch in Test_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break

print("TRAIN: ")
print(Train_IMG_Set.class_indices)
print(Train_IMG_Set.classes[0:5])
print(Train_IMG_Set.image_shape)
print("---"*20)
print("VALIDATION: ")
print(Validation_IMG_Set.class_indices)
print(Validation_IMG_Set.classes[0:5])
print(Validation_IMG_Set.image_shape)
print("---"*20)
print("TEST: ")
print(Test_IMG_Set.class_indices)
print(Test_IMG_Set.classes[0:5])
print(Test_IMG_Set.image_shape)


Model_Two = tf.keras.models.Sequential([
  # inputs 
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(256,)),
  # hiddens layers
  tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  # output layer
  tf.keras.layers.Dense(2,activation="softmax")
])
Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5,mode="min")
Model_Two.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
ANN_Model = Model_Two.fit(Train_IMG_Set,
                          validation_data=Validation_IMG_Set,
                          callbacks=Call_Back,
                      epochs=20)

#plt.plot(ANN_Model.history["accuracy"])
#plt.plot(ANN_Model.history["val_accuracy"])
#plt.ylabel("ACCURACY")
#plt.legend()
#plt.show()

Prediction_Two = Model_Two.predict(Test_IMG_Set)
Prediction_Two = Prediction_Two.argmax(axis=-1)
print(Prediction_Two)


import tkinter as tk
root= tk.Tk()


canvas1 = tk.Canvas(root, width = 1000, height = 1000)
canvas1.pack()

label1 = tk.Label(root, text='Fire Detection System', font=("Courier",40),bg='black',fg='white')
canvas1.create_window(450, 50, window=label1)
        
from PIL import ImageTk, Image  
from tkinter import filedialog
import os
#root.geometry("550x300+300+150")
#root.resizable(width=True, height=True)
import glob
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report

def upload_file():
    global img
    global filename
    global b2
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    image2=img.resize((300,300),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image2)
    b2 =tk.Label(root,image=img) # using Button 
    canvas1.create_window(250, 380, window=b2)

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#from tensorflow.keras.models import load_model

def prediction():
    img = image.load_img(filename,target_size=(256,256))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0) 
    Diff_Pred = Model_Two.predict(x)
    Diff_Pred = Diff_Pred.argmax(axis=-1)
    print(Diff_Pred)
    if Diff_Pred == 0:
        print('fire')
        label3 = tk.Label(root, text='Fire', font=("Courier",40),fg='red')
        canvas1.create_window(700, 600, window=label3)
    else:
        print('no fire')
        label4 = tk.Label(root, text='No Fire', font=("Courier",40),fg='red')
        canvas1.create_window(700, 600, window=label4)

btn1 = tk.Button(root, text='Upload Image/Video', command=upload_file, bg='orange',font=("Courier",20))
canvas1.create_window(620, 400, window=btn1)
btn2 = tk.Button(root, text='Detect', command=prediction, bg='green',font=("Courier",20))
canvas1.create_window(700, 200, window=btn2)

root.mainloop()
