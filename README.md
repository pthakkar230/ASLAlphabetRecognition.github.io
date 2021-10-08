# [ASL Alphabet Recognition Proposal](https://pthakkar230.github.io/ASLAlphabetRecognition.github.io/) 

## Introduction/Background

<img src="assets\asl diagram.png" height="500px">

American sign language is the primary method of communication between North Americans who are deaf or hard of hearing and the people around them. It is the 5th most common language in the United States. It is recommended that children born with conditions that affect their hearing be exposed to language as soon as possible. This requires that the affected children and their families learn the language as well. In addition, there is a need for those in the service industry such as educators, first responders, and caregivers to know this language as well. The most obvious point to start this process as with any person learning a new language is to understand the letters of the alphabet and how to spell common words.

There are a breadth of datasets that can be used to classify hand gestures as a particular letter in the English alphabet. We can leverage this to create a tool to train users to spell words in the English language, by taking a stream of images of the users’ hand gestures while spelling a certain word to verify correct spelling.

## Problem

The symbolic differences in the ASL alphabet can be difficult to distinguish between, particularly for beginners. We hope to enhance the speed and accuracy of the learning process by developing a program that generates real-time analysis of hand gestures, providing users with crucial feedback to augment their experience. We also hope that this project can be used for ASL speakers to communicate with those who are not familiar with the alphabet.

## Methods

Users will provide input from a camera source, which our program will use to extract relevant features.

<img src="assets\haar cascade.png" height="500px">

In order to achieve robust hand detection, we first need to identify the hand being manipulated within the frame and use image processing to generate a cleaner image. For this we will use a supervised learning method known as the Haar Cascade model to identify and extract portions of input frames with hands.

From here, we will use the Kaggle ASL dataset to train a convolutional neural network to classify the refined hand image and predict which symbol is being made to be displayed for the user.

## Results/Discussion

The classification of several different input images of people displaying ASL letters will serve as the supervised learning part of the project. The first step would be to ensure that we are correctly able to identify the features in images that correspond to a person’s hands. We then would want to measure the accuracy of our model’s recognition of ASL letters with varying parameters such as the race of the person in the input image, the size of the person’s hands, and various nuances in the angling of the person’s hands. 

Based on our initial testing accuracy obtained by training our model with the datasets we have identified, we might choose to expand our dataset in order to account for these varying conditions.


Discussion
The size of our dataset and the diversity of physical features in the images is important to use our model on a widespread basis. Machine learning models are prone to inherent biases based on the demographics of the team and the datasets, and we must be careful to identify these biases and expulge them. By doing so, we can ensure that we can identify and classify a person’s hands with varying factors in the input image. The research we perform to obtain an accurate model can lead to a promising feature allowing people to learn the ASL alphabet.
