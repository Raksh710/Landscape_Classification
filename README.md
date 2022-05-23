# landscape_classification
Given an input image, classify the image in the following category:  'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5.
&lt;br> &lt;/br> 
Above are the keys along with their tag (or value) are mentioned. A CNN model has been used with 3 Conv2D, 3 MaxPool2d, 1 Flatten, one dropout and 2 Dense layers.  &lt;br> &lt;/br>
After training the CNN model on 14034 images belonging to 6 classes, the CNN model was validated on a validation set with 3000 images belonging to 6 classes, on which an accuracy of 84.17% was achieved.  

Steps: 
1) Specify train, validation and test directory (where images are stored)
2) Use Image Generator to create more samples out of the given number of training samples (in order to detect the class more accurately). Images went through various processes like: zoomed in/out, sheared, rorated etc. 
3) Images from train and validation were subjected to the Image Generator created in step: 2. Note that in training the shuffle was True and that in validation it was False, because we want to keep the validation set in order to evalue the accuracy (which required the images to be in order
4) Image samples from train directory were fed to the CNN model and evaluated on the validation directory. 5) Image samples from test directory were also predicted and evaluated manually.
