#!/usr/bin/env python
# coding: utf-8

# ## Intel Image Classification Challenge: 
# Given an input image, classify the image in the following category: 
# 'buildings': 0,
# 'forest': 1,
# 'glacier': 2,
# 'mountain': 3,
# 'sea': 4,
# 'street': 5
# <br>
# </br>
# Above are the keys along with their tag (or value) are mentioned.
# A CNN model has been used with 3 Conv2D, 3 MaxPool2d, 1 Flatten, one dropout and 2 Dense layers. 
# <br>
# </br>
# After training the CNN model on 14034 images belonging to 6 classes, the CNN model was validated on a validation set with 3000 images belonging to 6 classes, on which an accuracy of 84.17% was achieved.
# 
# Steps:
# 1) Specify train, validation and test directory (where images are stored)
# 2) Use Image Generator to create more samples out of the given number of training samples (in order to detect the class more accurately). Images went through various processes like: zoomed in/out, sheared, rorated etc.
# 3) Images from train and validation were subjected to the Image Generator created in step: 2. Note that in training the shuffle was True and that in validation it was False, because we want to keep the validation set in order to evalue the accuracy (which required the images to be in order)
# 4) Image samples from train directory were fed to the CNN model and evaluated on the validation directory.
# 5) Image samples from test directory were also predicted and evaluated manually.

# In[1]:


# Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


from matplotlib.image import imread # imorting the function used to read a set of array as an image


# In[3]:


pwd # checking the present working directory


# In[6]:


my_dir = "C:\\Users\\HP\\Desktop\\Kaggle_Challenges\\Intel Image classification" # setting up a custom directory


# In[7]:


os.listdir(my_dir) # checking the elements of the custom directory


# In[10]:


train_dir = r"C:\Users\HP\Desktop\Kaggle_Challenges\Intel Image classification\seg_train\\seg_train" # setting the training data directory


# In[11]:


os.listdir(train_dir) # checking the elements training data directory for categories


# In[13]:


valid_dir = r"C:\Users\HP\Desktop\Kaggle_Challenges\Intel Image classification\seg_test\\seg_test" # setting the cross validation data directory


# In[14]:


os.listdir(valid_dir) # checking the elements cross-validaton data directory for categories


# In[64]:


test_dir = "C:\\Users\\HP\\Desktop\\Kaggle_Challenges\\Intel Image classification\\seg_pred" # setting the test data directory


# In[17]:


#os.listdir(test_dir)


# In[20]:


os.listdir(train_dir+ "\\buildings")[0] # checking the first image from training data directory


# In[25]:


img1 = imread(train_dir+ "\\buildings"+"\\0.jpg") # reading the image as an array


# In[28]:


plt.imshow(img1) # disploying the image


# In[29]:


img1.shape # checking the shape (lenght, breadth and color dimensions 'RGB')


# In[30]:


type(img1)


# In[31]:


img_shape = (150,150,3) # declaring the desired image shape


# In[32]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator # importing the function required to create more training samples


# In[33]:


# creating an image generator instance capable of modifying the image so that our model can get trained on more data samples
image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.20, height_shift_range=0.20, rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest' )
                               


# ### Creating a CNN model

# In[34]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


# In[35]:


model = Sequential() # creating a sequence instance

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=img_shape, activation='relu')) # creating a 2D convolution layer with the given parameters
model.add(MaxPooling2D(pool_size=(2, 2))) # creating a 2D Max pooling layer with the given parameters

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=img_shape, activation='relu')) # creating a 2D convolution layer with the given parameters
model.add(MaxPooling2D(pool_size=(2, 2))) # creating a 2D Max pooling layer with the given parameters

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=img_shape, activation='relu')) # creating a 2D convolution layer with the given parameters
model.add(MaxPooling2D(pool_size=(2, 2))) # creating a 2D Max pooling layer with the given parameters

model.add(Flatten()) # using a flattening layer to make the values 1D to feed forward the network into a normal Dense layer

model.add(Dense(1024,activation='relu')) # Using 1024 nodes in the normal dense layer using 'rectified linear unit' activation

model.add(Dropout(0.5)) # Using a dropout layer to randomly switch off 50% of the nodes of the previous layer to counter 'overfitting' 

model.add(Dense(6,activation='sigmoid')) # creating the output layer for 6 distinct categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) # compiling the CNN model


# In[36]:


model.summary() # checking the summary (in a way architecture) of the CNN model


# In[37]:


# Using early stopping to minimzie the chances of 'overfitting'
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,mode='min')


# In[38]:


batch_size = 16 # setting a batch size


# In[40]:


# Feeding in the image to the image genenration instance created above (shuffle is True as we want to 'randomize' our training)
train_image_gen = image_gen.flow_from_directory(train_dir,target_size=img_shape[:2],color_mode='rgb', batch_size=batch_size,class_mode='categorical', shuffle=True)                                       


# In[42]:


# Feeding in the image to the image genenration instance created above (shuffle is False as we want the validation samples to be in the given order)
valid_image_gen = image_gen.flow_from_directory(valid_dir,target_size=img_shape[:2],color_mode='rgb', batch_size=batch_size,class_mode='categorical', shuffle=False)                                       


# In[43]:


train_image_gen.class_indices


# In[44]:


valid_image_gen.class_indices


# #### Fitting the model on training image generator (fed from training data directory) and validating it on validation image generator (fed from validation data directory)

# In[45]:


#results = model.fit_generator(train_image_gen,epochs=30, validation_data=valid_image_gen,callbacks=[early_stop])


# In[46]:


# saving the model
#model.save("keras_model.h5")


# In[110]:


from tensorflow.keras.models import load_model


# In[111]:





# In[112]:


# loading the model
model = load_model("keras_model.h5") 


# In[47]:


# making a dataframe to account for losses (both training and validation) and accuracy (both training and validation) for each epoch for the above created CNN model
losses_df = pd.DataFrame(model.history.history) 


# In[48]:


losses_df.head() # checking the head of the dataframe created above


# In[49]:


losses_df.plot(figsize=(12,6))


# #### Predicting and evaluating the CNN model for samples in validation data directory

# In[50]:


cnn_pred = model.predict_generator(valid_image_gen) 


# In[51]:


cnn_pred


# #### for each prediction choosing only that category which has the highest probability. 
# #### Our CNN model outputs the probabilty (for every fed image) regarding all the 6 different categories.
# #### Whichever category has the highest probability is finalized as the image class. E.g for the first prediction, the class '0' or 'buildings' has the highest probability. 
# #### Hence, we classify the image as an image belonging to the class "buildings" 

# In[52]:


cnn_pred_actual = np.argmax(cnn_pred,axis=1) # returining the class which has the highest probability.


# In[116]:


cnn_pred_actual


# In[54]:


set(cnn_pred_actual) # checking the distinct values


# #### Evalauting the model (84.16% accuracy)

# In[55]:


from sklearn.metrics import confusion_matrix,classification_report, accuracy_score


# In[57]:


print(classification_report(valid_image_gen.classes, cnn_pred_actual), end='\n')
print(confusion_matrix(valid_image_gen.classes, cnn_pred_actual), end='\n')
print(accuracy_score(valid_image_gen.classes, cnn_pred_actual), end='\n')


# In[58]:


valid_image_gen.class_indices


# #### Least precision and recall for detecting mountain. Thus, the model has slight difficulty in recognizing 'Mountains'.

# In[66]:


## Taking the image from test directory. Shuffle is False as we want the test image to be in order
test_image_gen = image_gen.flow_from_directory(test_dir,target_size=img_shape[:2],color_mode='rgb', batch_size=batch_size,class_mode='categorical', shuffle=False)                                             


# In[67]:


cnn_pred_test_predictions = np.argmax(model.predict_generator(test_image_gen), axis=1) # returining the class with highest probability for test predictions


# In[68]:


cnn_pred_test_predictions


# ## Let's do some single random predictions

# In[76]:


from tensorflow.keras.preprocessing import image


# In[72]:


os.listdir(test_dir+"\\seg_pred")[:5]


# In[73]:


guess_img_1 = imread(test_dir+"\\seg_pred\\10004.jpg")


# In[74]:


plt.imshow(guess_img_1) # street:


# In[78]:


sample_img_1 = image.load_img(test_dir+"\\seg_pred\\10004.jpg" ,target_size=(150,150)) # loading the image


# In[79]:


sample_img_1 = image.img_to_array(sample_img_1) # converting the image file into numpy array


# In[80]:


sample_img_1 = np.expand_dims(sample_img_1,axis=0) # expaning dimensions on axis=0 or vertically to match the required dimensions


# In[81]:


sample_img_1 = sample_img_1/255 # scaling the image


# In[88]:


random_pred = np.argmax(model.predict(sample_img_1), axis=1) # doing prediction and returning the class with highest probability


# In[95]:


random_pred # results. Image is dominates by Street (although it has buildings as well)


# In[93]:


valid_image_gen.class_indices


# ### Let's make a function which just takes in image path and gives us the results

# In[103]:


key_values = list(valid_image_gen.class_indices.keys()) # making list of key values or categories
answers = list(valid_image_gen.class_indices.values()) # making list of index values of categories


# In[104]:


key_values[random_pred[0]]


# In[107]:


def image_labeller(image_path):
    '''
    This function displayes the image and then uses the CNN model created above to classify the image and then prints out the result.
    '''
    plt.imshow(imread(image_path)) # reading the image file
    plt.show() # displaying the image
    sample_img = image.load_img(image_path ,target_size=(150,150)) # loading the image
    sample_img = image.img_to_array(sample_img)  # converting the image file into numpy array
    sample_img = np.expand_dims(sample_img,axis=0) # expaning dimensions on axis=0 or vertically to match the required dimensions
    sample_img = sample_img/255 # scaling the image
    
    img_pred = np.argmax(model.predict(sample_img),axis=1) # doing prediction and returning the class with highest probability
    print("\n")
    print(f"This image belongs to the class: {key_values[img_pred[0]]}")


# In[108]:


image_labeller(test_dir+"\\seg_pred\\10017.jpg") # calling the above created function on a random image


# #### Correct prediction

# In[ ]:





# In[109]:


image_labeller(test_dir+"\\seg_pred\\3.jpg") # calling the above created function on a random image


# #### Correct prediction

# In[ ]:





# In[114]:


image_labeller(test_dir+"\\seg_pred\\129.jpg") # calling the above created function on a random image


# #### Correct Prediction

# In[117]:


image_labeller(r"C:\Users\HP\Desktop\Kaggle_Challenges\Intel Image classification\My_img.jfif") # calling the above created function on a random image


# #### Incorrect prediction:
# This could be because of the fact that the model was trained on simple images with only minor overlapping of categories. However, this image is quite complex as shows skyline of buildings, sea and a part of it was frozen (similar to glacier because the model might have taken the distant buildings as mountains).

# In[ ]:




