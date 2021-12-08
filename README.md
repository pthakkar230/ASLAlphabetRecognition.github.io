# [ASL Alphabet Recognition Report](https://pthakkar230.github.io/ASLAlphabetRecognition.github.io/) 


## Introduction/Background

American sign language is the primary method of communication between North Americans who are deaf or hard of hearing and the people around them. It is the 5th most common language in the United States. It is recommended that children born with conditions that affect their hearing be exposed to language as soon as possible. This requires that the affected children and their families learn the language as well. In addition, there is a need for those in the service industry such as educators, first responders, and caregivers to know this language as well. The most obvious point to start this process as with any person learning a new language is to understand the letters of the alphabet and how to spell common words.

There are a breadth of datasets that can be used to classify hand gestures as a particular letter in the English alphabet. We can leverage this to create a tool to train users to spell words in the English language, by taking a stream of images of the users’ hand gestures while spelling a certain word to verify correct spelling.

## Problem

The symbolic differences in the ASL alphabet can be difficult to distinguish between, particularly for beginners. We hope to enhance the speed and accuracy of the learning process by developing a program that generates real-time analysis of hand gestures, providing users with crucial feedback to augment their experience. We also hope that this project can be used for ASL speakers to communicate with those who are not familiar with the alphabet.

## Data Collection

XML-based hand detection models were downloaded from various sources on the internet. This is the chosen format for OpenCV haar cascades. We have several models for different hand views (palm, fist, side, etc.) For the next phase of the project we will explore creating our own XML models.

We recorded our own samples as well as used a Kaggle hand dataset in order to complete our data. The hand dataset from Kaggle has a total of 13050 hand instances with annotations attached to each image. Any hand that exceeds the threshold of 100 x 100 pixels in a box shape are detected as hands and are used for evaluation. The dataset claims to have collected data with no bias for the visibility of people, environment, or any other constraint of type of image as long as a person was present. The annotations come in the form of a bounding rectangle on qualifying image, and it is not required to be axis dependent, or have to be oriented with respect to a wrist. 

## Methods

### Object Detection + Image Preprocessing

As outlined in our initial project debrief, the general framework for our system depends on robust hand detection. We have chosen to use the Haar Cascade model to identify and extract portions of input frames with hands to standardize and then feed into a convolutional neural net for classification.

### Classification

We also began work on our CNN model, using training data from our dataset. In the next phase of our project we will also begin feeding the output of the haar detection model into this CNN to build the full data pipeline from raw image to gesture recognition, but for now we stuck to using the existing labelled data so that classification and object detection could be worked on simultaneously. We are currently using an existing architecture, the MobileNet architecture, due to its lightweight size which allows us to train faster and produce better early-stage results.

### Word Spelling + Letter Stitching

For the final phase of the project, we plan to implement live video feed logic to extract words on-the-fly as the user spells them out. This will not be covered in this midterm report.

## Results and Discussion

### Object Detection

Hand detection presents significant challenges. The many distinct poses in ASL significantly distort the appearance of the hand, obstructing the view of fingers and other unique visual elements. Thus, it is crucial to have robust models for several key positions in the language. We have identified three to focus on for the scope of this project: palm, fist, and side.

Unfortunately, the XML Haar Cascades models we found online are not very accurate at detecting these three broad categories. The closed fist is the easiest to identify, but also identifies significant amounts of various artifacts in test footage. The following images represent the sample space selected from a single frame:

<img src="assets\image4.jpg" height="100px">
<img src="assets\image5.jpg" height="100px">
<img src="assets\image6.jpg" height="100px">
<img src="assets\image8.jpg" height="100px">
<img src="assets\image9.jpg" height="100px">

This represents a best-case scenario, as the closed fist has shown to be the most consistently accurate model from our designated XMLs. Other models perform significantly worse, generating up to 2-3x the artifacts per frame.

Going forward we will look into training our own XMLs for more robust detection across the three categories.

### Image Preprocessing

The haar-cascade model actually does most of the preprocessing of our data for us, as the algorithm includes edge and feature detection. Using a rough heuristic of bounding box size, we then eliminate obvious artifacts. It should be noted that this does not eliminate all artifacts. With the remaining detection boxes, we then crop and normalize to use as input for our classification algorithm.

### Classification

The classification task proved to be a bit more successful. Our results from our CNN after training for 2 epochs are as follows. 

<img src="assets\image1.png" height="250px">
<img src="assets\image7.png" height="250px">

Our initial results are promising, with a peak validation accuracy of 75%. As you can see from the decrease in validation accuracy with a corresponding increase in training accuracy, our CNN model currently suffers badly from overfitting. The same trend is observed in our cross-entropy loss, with loss decreasing on the training data but increasing in our validation set. This can be addressed by introducing regularization and dropout layers, both of which we plan on experimenting with in the next stage of our project. We also would like to train for more epochs once we address overfitting. We also plan on experimenting with our own architectures other than the MobileNet using a combination of densely connected, convolution, drop out, and batch normalization layers.




### Introduction/Background
American sign language is the primary method of communication between North Americans who are deaf or hard of hearing and the people around them. It is the 5th most common language in the United States. It is recommended that children born with conditions that affect their hearing be exposed to language as soon as possible. This requires that the affected children and their families learn the language as well. In addition, there is a need for those in the service industry such as educators, first responders, and caregivers to know this language as well. The most obvious point to start this process as with any person learning a new language is to understand the letters of the alphabet and how to spell common words.

There are a breadth of datasets that can be used to classify hand gestures as a particular letter in the English alphabet. We can leverage this to create a tool to train users to spell words in the English language, by taking a stream of images of the users’ hand gestures while spelling a certain word to verify correct spelling.


### Problem

The symbolic differences in the ASL alphabet can be difficult to distinguish between, particularly for beginners. We hope to enhance the speed and accuracy of the learning process by developing a program that generates real-time analysis of hand gestures, providing users with crucial feedback to augment their experience. We also hope that this project can be used for ASL speakers to communicate with those who are not familiar with the alphabet.


### Data Collection

Our initial object detection datasets consisted of XML-based hand detection models that were downloaded from various sources on the internet. This is the chosen format for OpenCV’s implementation of Haar cascades. We have several models for different hand views (palm, fist, side, etc.).

For object detection we ultimately decided to use the Egohands dataset, which consists of 4800 frames including 15053 high-quality, pixel-level segmentations of labeled hands. This data was collected from the first-person perspective of users wearing Google Glasses engaging in a multitude of unconstrained hand poses. Although this dataset was initially developed for use with MATLAB, it is compatible with Python.

We recorded our own samples as well as used a Kaggle hand dataset in order to complete our convolutional neural net data. The hand dataset from Kaggle has a total of 13050 hand instances with annotations attached to each image. Any hand that exceeds the threshold of 100 x 100 pixels in a box shape are detected as hands and are used for evaluation. The dataset claims to have collected data with no bias for the visibility of people, environment, or any other constraint of type of image as long as a person was present. The annotations come in the form of a bounding rectangle on qualifying image, and it is not required to be axis dependent, or have to be oriented with respect to a wrist. 


### Methods

#### Object Detection + Image Preprocessing

##### Haar Cascade

As outlined in our initial project debrief, the general framework for our system depends on robust hand detection to identify and extract portions of input hands to standardize into images that can be fed into a convolutional neural network for classification. It should be noted that our CNN model does not require significant preprocessing, only a size requirement of 224x224x3, so significant transformations or other feature extraction algorithms like edge detection are unneeded.

We initially chose to use the Haar Cascade model. Haar Cascade is a classification model that relies on a binary dataset. For example, as we were dealing with hands, we would train our model on a variety of images with hands and a variety of images that are not hands at all. It relies on edge detection specifically to attempt to classify the desired feature in the input image. Haar Cascade will attempt to classify pixels in an input image from 0 to 1 by use of pixel intensity and layout of the pixels. If a certain region of pixels has an average difference close to 1 than the pixels around it, it is likely that the model has classified a region of interest. Even though Haar Cascade models are one of the oldest machine learning tools used for feature detection, they are prone to false-positives making them slightly difficult to use for our use case. 

##### Single Shot Detection 

After less than ideal results with Haar Cascade, we moved to Single Shot Detection (SSD). SSD architecture depends on a base network, e.g. a CNN, and multiple convolutional layers. The base network extracts features to find fixed-size bounding boxes in an image and their corresponding probabilities for a designated object. It should be noted that the part of the CNN used in SSD does not include classification layers. The convolutional layers decrease in size to allow for size-invariant detection. The base network is trained with the Egohands dataset.




SSD two-tier architecture (https://arxiv.org/pdf/1512.02325.pdf)

By applying convolutional filters to CNN feature maps, unlike other object detection approaches which rely on bounding box and sliding window, SSD architecture is comparatively performant which allows for very fast real-time processing.

#### Classification

For this stage of our project, we are using transfer learning by attempting to apply pre trained CNN models to our dataset. This is because the task of image classification is a well-studied problem in the field of machine learning, and many existing models have been created to classify images such as the CIFAR-10 (images of objects, cars, animals, etc.) or the MNIST (hand written number classification) dataset. All of these models were initialized with the weights for the ‘imagenet’ dataset as it has a broad range of categories and classifications (1000+) and hopefully will apply to our application well. Therefore, by using their model as a starting point, we hoped to achieve better results with our model than if we had started training the model from scratch.

We are currently using an existing architecture, the MobileNet architecture, with cross entropy loss and average pooling due to the intended real-time nature of this project. The lightweight size of the MobileNet architecture allows us to produce a result with less delay. The model takes in a 224 x 224 x 3 image (as that is the standard dimension of the dataset), which we produce by downsampling and cropping.

The architecture of the MobileNet is as follows:

https://arxiv.org/pdf/1704.04861.pdf
The main reason why the MobileNet architecture is more computationally lightweight than most convolutional neural networks is its usage of a “depthwise separable convolution.” The depthwise separable convolution can be factorized into 2 separate convolutions, one for filtering (3x3 depthwise) and one for combining features (1x1 conv). This reduces the complexity of both training and execution.

Since we used a pretrained network that produces 1000 feature output, our model also has a fully connected layer after MobileNet that reduces the 1000 features to the 26 letters of the ASL alphabet.

The other architecture we used was the ResNet50 architecture, which was shown to produce very good results with the CIFAR 10 dataset for classification. ResNet50 is not lightweight by any means, it is composed of 50 convolutional layers, with the filters in each layer increasing the deeper in the network. The idea behind the layers is that the deeper you go the more complex features are filtered and detected based on smaller and simpler features. However, we were unable to get good results with ResNet50 (see results), and the computation time of ResNet50 (5+ seconds) was unacceptable for our application.

https://pytorch.org/hub/pytorch_vision_resnet/
Note that there is an average pooling and softmax layer at the end of this network not pictured.

Word Spelling + Letter Stitching


### Results and Discussion

#### Object Detection

Hand detection presents significant challenges. The many distinct poses in ASL significantly distort the appearance of the hand, obstructing the view of fingers and other unique visual elements.

##### Haar Cascade

Through our testing on real-world data, we found accuracy to be significantly less than ideal.  One of the drawbacks of using Haar Cascade is that it requires the use of multiple models: palm, fist, and side view. Unfortunately, the XML Haar Cascades models we found online are not very accurate at detecting these three broad categories. The closed fist is the easiest to identify, but also identifies significant amounts of various artifacts in test footage. The following images represent the sample space selected from a single frame:


This represents a best-case scenario, as the closed fist has shown to be the most consistently accurate model from our designated XMLs. Other models perform significantly worse, generating up to 2-3x the artifacts per frame. It became very apparent through initial testing that we needed a more robust method of object detection.

##### Single Shot Detection (SSD)

Single Shot Detection proved to be significantly more accurate and performant than Haar Cascade. It struggles most in recognizing certain hand positions (namely those with outstretched fingers or from the side), but if you leave the hand in the frame for long enough with a relatively clean background the model generally picks up on it within a few seconds, even in these more difficult positions. You can see in our sample video the clear improvement over Haar Cascade. It proves to be very robust in detecting the hand across the palm, fist, and side view.

##### Image Preprocessing
Using a rough heuristic of the bounding box for the detected hand in the frame, we then scale the image to 224x224 to comply with our classification network specifications.

#### Classification
As mentioned in the methods section, our results with the other architectures were unusable. Here is the following result for ResNet50:

As you can see, the peak validation accuracy of ResNet50 with default parameters was about 20%. While it is important to note that while the hyperparameters were not tuned, this stood in stark comparison to our alternative architecture MobileNet with default parameters. In addition, one forward pass through ResNet50 took about 5 seconds on our computers, which was unacceptable for a real-time application. For these reasons, we decided not to pursue experimentation down this path, either through tuning hyperparameters to increase accuracy, or using a smaller version of the model (such as the 19 or 34 layer versions) to increase performance.

However, our initial results with MobileNet were promising, with an initial validation accuracy of around 70%.



Initially, we thought the decrease in validation accuracy with a corresponding increase in training accuracy was an indication of overfitting. However, upon running for more epochs, this trend was not observed, and our initial results must have been an anomaly. 

Therefore, we focused on tuning hyperparameters. We found our model performed best when we changed our initial learning rate from the default of .005 to .05.


We also were able to get better results when dropout layers were introduced with a probability of .001, a phenomenon that we couldn’t quite understand as our model clearly didn’t suffer from overfitting at this point. I believe that our model must have been hitting some kind of local minima, and that when the dropout layers were introduced this allowed the model to explore different avenues to find an even better minima. This can also be seen in the brief drop in validation and training loss and accuracy in epoch 3, only to find a better minima by epoch 5.


When we attempted to connect the parts of our pipeline together, we saw a significant drop in classification accuracy. While we initially expected a drop in accuracy because of the cleanliness of our data (which was shot with a blank background) and our webcam footage, we noted that a large drop in classification accuracy occurred when we had to downgrade our tensorflow version from 2.0 to 1.0. We were unable to increase our validation accuracy past 30% despite having the exact same architecture and being trained on the exact same data. 



### Conclusion

Our final pipeline does demonstrate significant difficulty and has a heavy bias towards certain letters. For example, letters ‘a’, ‘b’, and ‘w’ are detected without significant issue, but oftentimes other letters symbols will be conflated with those symbols, e.g. other fist based symbols such as ‘s’ or ‘t’ are detected as ‘a’.

The pipeline has much room to improve, and our initial project idea was very ambitious. In addition to the errors in classification, there is also some difficulty with hand detection. The model often does not detect the hand despite being very visible in the frame, and other times the bounding box often cuts out crucial parts of the hand. This makes the classification of certain symbols like ‘d’ very difficult, where the finger is often cut out of the frame and the classification model doesn’t have access to crucial features to detect the symbol as a ‘d’.
To address these problems, if we had more time we could explore the use of masking techniques as a preprocessing step. This would allow for background noise to be removed entirely from the image, and would help us understand where our detection model thinks the hand is and why it would crop out certain important features.

