# BrainTumor_CNN_Model
## About
---
## Table of Contents
---
## Introduction to CNN
&nbsp;&nbsp;&nbsp;
# explain metrices used (recall), validation data
---
## 
* [Google Colab](https://colab.research.google.com/github/jlee92603/BrainTumor_CNN_Model/blob/main/BrainTumor_CNN_Model.ipynb)

## Getting Started
This convolutional neural network is coded with the Python's Tensorflow Keras package on Google Colab. 

### Installations
Important libraries to install are:
* google-api-python-client
* opencv-python-headless
* keras-metrics
* keras-self-attention

### Downloading Dataset
The brain tumor dataset is acquired from Kaggle [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). This dataset contains 7023 images of human brain MRI images that are classified into 4 classes: 
* glioma: a brain tumor that forms when glial cells, cells that support the central nervous system and nerves, grow out of control
* meningioma: a brain tumor that resides primarily in the central nervous system
* no tumor: the absense of brain tumor
* pituitary: a brain tumor that is classified as the unusual growth that develops in the pituitary gland
This dataset contains a set of testing data files and training data files. 

### Connecting Drive and GPU
The dataset is downloaded and uploaded on to Google Drive, which is connected to the Colab notebook. Additionally, Google Colab's T4 GPU is connected for faster model fitting. 

### Importing Libraries
The following packages are imported: 
```
import os
from os import listdir
from tabulate import tabulate
import numpy as np
import math
import cv2
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, Activation, MaxPooling2D, Conv2D, Attention, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Model, Sequential
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam, SGD, AdamW, Adamax, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.losses import CategoricalFocalCrossentropy, CategoricalHinge, SparseCategoricalCrossentropy
import seaborn as sn
from scipy import interp
from itertools import cycle
import sklearn
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
```

### Accessing and Reading Image Data
The data is put into categorized folders for each of the types of tumor/noTumor; hence, the image files from multiple folders are loaded and put into lists for the name of the image file and the classification label of the image. 
```
# data path from drive
training_data_path = filepath + 'brain tumor image files/archive/Training'
testing_data_path = filepath + 'brain tumor image files/archive/Testing'

# load the file names of the images in multiple folders
def load_images_files_from_folder(directory):
    image_files = []
    file_labels = []

    # for each folder in directory
    for folder in os.listdir(directory):
      # for each file in each folder in directory
      for file in os.listdir(os.path.join(directory,folder)):
        # add file if it ends with .jpg
        if file.endswith(".jpg"):
            file_name = os.path.join(directory, folder, file)
            if image_files is not None:
                image_files.append(file_name)
                file_labels.append(folder)

    # returns tumor image file names and types of tumor they are
    return image_files, file_labels

# access images from multiple folders
image_files, file_labels = load_images_files_from_folder(training_data_path)
test_image_files, test_file_labels = load_images_files_from_folder(testing_data_path)

# display couple images
fig, ax = plt.subplots(3,4)
for x in range(12):
  img = cv2.imread(image_files[x*430],1)
  ax[int(x/4),int(x%4)].set_title(os.path.basename(os.path.dirname(image_files[x*430])))
  ax[int(x/4),int(x%4)].imshow(img)
  ax[int(x/4),int(x%4)].axis("off")
plt.show()

# format information in a table
header = ["#", "image file name", "tumor type"]
data = []

for ct, ele in enumerate(image_files[0:3]):
  data.append([ct, os.path.basename(ele), os.path.basename(os.path.dirname(ele))])

for ct, ele in enumerate(image_files[2000:2003]):
  data.append([ct+3, os.path.basename(ele), os.path.basename(os.path.dirname(ele))])

for ct, ele in enumerate(image_files[3000:3003]):
  data.append([ct+6, os.path.basename(ele), os.path.basename(os.path.dirname(ele))])

for ct, ele in enumerate(image_files[5000:5003]):
  data.append([ct+9, os.path.basename(ele), os.path.basename(os.path.dirname(ele))])

print(tabulate(data, header, tablefmt="grid"))
```
A couple of the images are displayed with their classification label: 
<img width="345" alt="Screen Shot 2023-10-30 at 11 38 43 PM" src="https://github.com/jlee92603/BrainTumor_CNN_Model/assets/70551445/8bfa30bb-34cc-453a-a82c-4ea44028f86a">

Additionally, a table is created to display a couple of the image file names and their type of tumor:
<img width="245" alt="Screen Shot 2023-10-30 at 11 39 28 PM" src="https://github.com/jlee92603/BrainTumor_CNN_Model/assets/70551445/0a99591d-32cb-40c8-a5f9-831a639cb507">

### Splitting and Concatenating Data
The data is concatenated as a data frame. Additionally, training data set is split into validation data and training data. Then, ImageDataGenerator is used to ensure that the model receives new variations of images at each epoch. 
```
# data concatenation as data frame
def concatenateDataAsDF(image_files, image_file_labels):
  Fseries = pd.Series(image_files , name = 'filepaths')
  Lseries = pd.Series(image_file_labels , name = 'label')
  return pd.concat([Fseries , Lseries] , axis = 1)

# concatenate training data as data frame
train_df = concatenateDataAsDF(image_files, file_labels)
train_df # training data as a data frame

# concatenate testing data as data frame
test_df = concatenateDataAsDF(test_image_files,test_file_labels)
test_df # testing data as a data frame

# separate validation data from training data
valid_data, train_data = train_test_split(train_df, train_size=0.3, shuffle=True, random_state=42)

# ImageDataGenerator (generate tensor image data)
img_size = (224 ,224) # size of image
batch_size = 4 # ImageDataGenerator produces 4 images each iteration of training
tr_gen = ImageDataGenerator()
ts_gen= ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_data , x_col = 'filepaths' , y_col = 'label' , target_size = img_size , class_mode = 'categorical' , color_mode = 'grayscale' , shuffle = True , batch_size =batch_size)

valid_gen = tr_gen.flow_from_dataframe(valid_data , x_col = 'filepaths' , y_col = 'label' , target_size = img_size , class_mode = 'categorical',color_mode = 'grayscale' , shuffle= True, batch_size = batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df , x_col= 'filepaths' , y_col = 'label' , target_size = img_size , class_mode = 'categorical' , color_mode= 'grayscale' , shuffle = False , batch_size = batch_size)
```

## CNN Modeling
A convolutional neural network model requires several layers:
* The convolutional layer (Conv2D), which convoles the input by moving filters along the input vertically and horizontally, and computing the dot product of the weights and inputs with biases.
* The max pooling layer is an operation that calculates the max value for patches of feature maps to create a downsampled (pooled) feature map. Its purpose is to reduce dimensions while preserving the most relevant features in a local region, reduce overfitting, and increase computational efficiency.
* The flattening layer is to put the data into a 1D array for input into the next layer.
* The denase layer contains all neurons that are deeply connected within themselves. It is useful for when associations can exist among any feature to any other feature in data point.

A convolutional layer has many different parameter inputs:
* filters: the dimension of output space. It is a feature detector, which is important in identifying various features.
* kernel size: the kernal size is the convolution window used. Increasing the kernal size allows the model to capture more global features from the input image.
* padding: adds extra pixels around the input image to prevent spatical shrinking and making sure the edges are not lost.
* activation function: ReLU is used for the network to learn non-linear relationships between the input and ouput
* input shape (rows, cols, channel): 1 channel for grayscale images, 3 channels for rgb.

Another relevant term is epoch, which is the number of times the learning algorithm will work through the entire training dataset. In this model, callbacks is used during the model fitting process to save the model of the epoch with the best validation data recall value.

### Creating Model
```
# create the convolutional neural network model
def createModel(filters=[16,32,64]):

  # initialize CNN
  model = Sequential() # Sequential Model consists of sequence of layers

  # add first convolution layer
  model.add(Conv2D(filters=filters[0],kernel_size=2,padding="same",activation="relu",input_shape=(224,224,1)))

  # max pooling operation
  model.add(MaxPooling2D(pool_size=2))

  # add more layers (Conv2D and MaxPool)
  for filter in filters[1:]:
    model.add(Conv2D(filters=filter,kernel_size=2,padding="same",activation ="relu"))
    model.add(MaxPooling2D(pool_size=2))

  # flattening operation (convert data into 1D array for input into next layer)
  model.add(Flatten())

  # fully connected layer and ouput layer; input units: dimension of output
  model.add(Dense(500,activation="relu")) # 250; 150 # variation of num of filters
  model.add(Dense(4,activation="softmax")) # 4 categories

  # softmax is similar to sigmoid; has smaller variance than relu
  (model.summary())

  return model
```
### Compiling Model
When compiling the model, the metrices Recall is used. Recall metrics focuses on capturing as many true positives and are usually used when the false negatives are costly. In this case, true positives are the result of the model predicting the existence of brain tumor when the patient actually has brain tumor, while false negatives are prediction that there is no brain tumor when the patient actually has brain tumor. The false negatives, in this scenario, are costly because predicting that the patient does not have tumor when they actuall do lets the tumor go unnoticed and untreated. Hence, we want to maximize the Recall value.
```
# compile CNN model
def compileModel(model, filepath, optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                 metrics=[keras.metrics.Recall()], shuffle=False):

  # loss function
  model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

  # model check point
  checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_recall', verbose=1,
                              save_best_only=True, mode='max')

  # fit model on training set
  return model.fit(x=train_gen, epochs=10, callbacks=checkpoint, verbose=1, validation_data=valid_gen, validation_steps=None, shuffle=shuffle)
```
### Evaluating the Performance
To analyze the results and evaluate the performance of the model, the model of the epoch with the highest validation data recall is tested on the testing data. The losses and recall graphs for the training and validation data is graphed for each epoch. The confusion matrix for the testing data predictions are also graphed to evaluate model performance. 
```
# test model of best epoch on testing data
def testModel(filepath, test_data=test_gen):
  bestModel = keras.models.load_model(filepath)
  pred = bestModel.predict(test_data)
  y_pred = np.argmax(pred, axis=1)
  return y_pred

# plot the losses and recall for the training and validation data
def plotResults(cnn_model, saveAs=False, filepath=filepath):
  fig, ax = plt.subplots(1,2)

  # plot loss graph
  ax[0].set_title("Loss Graph")
  ax[0].plot(cnn_model.history['loss'], label='train loss')
  ax[0].plot(cnn_model.history['val_loss'], label='val loss')
  ax[0].legend()

  # plot recall graph
  ax[1].set_title("Recall Graph")
  ax[1].plot(cnn_model.history['recall'], label='train recall')
  ax[1].plot(cnn_model.history['val_recall'], label='val recall')
  ax[1].legend()

  plt.tight_layout()
  if saveAs!=False:
    plt.savefig(filepath + saveAs)

  plt.show()

# create and plot confusion matrix
def createConfusionMatrix(y_pred, saveAs=False, filepath=filepath, test_gen=test_gen):
  # create confusion matrix with sklearn.metrics
  # classes is the actual values 0,1,2,3 corresponding to each type of tumor
  conf_mat = sklearn.metrics.confusion_matrix(test_gen.classes, y_pred)

  # use seaborn heatmap to plot confusion matrix
  # change resulting confusion matrix to data frame
  confusion_matrix_df = pd.DataFrame(conf_mat)

  sn.set(font_scale=1)
  cm_plot = sn.heatmap(confusion_matrix_df, annot=True, fmt="1.0f", linewidth=0.5,
                                cmap=sn.cubehelix_palette(as_cmap=True))

  cm_plot.set_xticklabels(list(test_gen.class_indices.keys()), rotation=45)
  cm_plot.set_yticklabels(list(test_gen.class_indices.keys()), rotation=0)

  plt.title('Confusion Matrix')
  plt.ylabel('Actual Values')
  plt.xlabel('Predicted Values')

  if saveAs!=False:
    plt.savefig(filepath + saveAs, bbox_inches='tight')

  plt.show()
```
### Clear Current Model
# CLEAR MODEL SESSION
```
keras.backend.clear_session()
```
### Gather Results from Multiple Models
The results of the models with different hyperparameters are saved and compared at the end. 
```
# create empty lists to store all the loss/recall graphs and confusion matrices
recall_graphs = []
conf_matrices = []

# create empty list to store the results of each model
results = [] # [cohort, best recall, optimizer, learning_rate, filters, other]
heading = ['Cohort','Best Recall Values', 'Optimizer', 'Learning Rate', 'Filters', 'Other']

# function to plot all the graphs collected from each model
def plotAllGraphs(conf_matrices, recall_graphs, filepath=filepath):
  plt.figure(figsize=(20,80))
  row = len(conf_matrices)
  col = 2

  for i in range(row*col):
    plt.subplot(row, col, i+1)
    plt.axis('off')
    if i%2!=0:
      plt.title(conf_matrices[i//2],fontsize=20)
      plt.imshow(mpimg.imread(filepath+conf_matrices[i//2]+'.png'))
    else:
      plt.title(recall_graphs[i//2],fontsize=20)
      plt.imshow(mpimg.imread(filepath+recall_graphs[i//2]+'.png'))
```
---
## Types of Hyperparameters Used
The functions above are used to create and compile the model, test the model on the testing data, evalute the performance of the model. 
```
# CREATE AND COMPILE MODEL
# create model # filters=[16,32,64]
model = createModel()

# compile model # loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001)
cnn_model = compileModel(model, filepath, optimizer=Adam(learning_rate=0.001))

# TEST MODEL ON TESTING DATA
y_pred = testModel(filepath)

# EVALUATE AND SAVE RESULTS
# evaluate results
plotResults(cnn_model, saveAs='Cohort A Recall and Loss')
createConfusionMatrix(y_pred, saveAs='Cohort A Confusion Matrix')

# save results
conf_matrices.append('Cohort A Confusion Matrix')
recall_graphs.append('Cohort A Recall and Loss')

highest_recall = max(cnn_model.history['val_recall'])
results.append(['A', highest_recall, 'Adam', 0.001, 'categorical crossentropy', '16, 32, 64', None])

# CLEAR MODEL SESSION
keras.backend.clear_session()
```
Grid search is used to find the optimal hyperparameters that create the model with the best performance, or highest recall value. The cohorts below are the different hyperparamters used: 
* Cohort A: Adam optimizer with 0.001 learning rate
* Cohort B: Adam optimizer with 0.01 learning rate
* Cohort C: Adam optimizer with 0.0001 learning rate
* Cohort D: Adam optimizer with the filters all set as 32
* Cohort E: Adam optimizer with another set of Conv2D and MaxPool2D layers
* Cohort F: AdamW optimizer with 0.001 learning rate
* Cohort G: SGD optimizer with 0.001 learning rate
* Cohort H: Adamax optimizer with 0.001 learning rate
* Cohort I: Adamax optimizer with 0.0001 learning rate
* Cohort J: AdamW optimizer with 0.0001 learning rate
* Cohort K: Adamax optimizer with 0.005 learning rate
* Cohort N: Adam optimizer with 0.001 learning rate and an attention layer
* Cohort O: Adam optimizer with 0.0001 learning rate and shuffling between each epoch
* Cohort Z: EfficientNetB3 architecture (a model introduced in the paper EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan and Quoc V. Le) 

## Model Results and Comparison
The validation data recall values for each cohort are as follows: 
<img width="564" alt="Screen Shot 2023-10-31 at 12 07 10 AM" src="https://github.com/jlee92603/BrainTumor_CNN_Model/assets/70551445/547e6eaa-9767-44fd-bf89-232e85b0aa44">




## Conclusion

---
