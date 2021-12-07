# [ASL Alphabet Recognition Midpoint](https://pthakkar230.github.io/ASLAlphabetRecognition.github.io/) 


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


