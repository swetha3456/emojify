# emojify

## Project Overview
This project aims to use ML models to convert facial expressions into emojis.

## Project Goal
- To build an interface that makes it easy for users to interact with emojis without having to scroll through a long list.
- To incorporate as many emojis as possible into the classifier

## Installation
1. Install the following Python libraries:
- OpenCV
- MediaPipe
- scikit-learn
2. Clone the repository. 
3. Download the dataset linked below and save the images folder in the main repository.
4. Run train_model.py, followed by emojify.py.
5. Try out various facial expressions and hand gestures to see the corresponding emojis displayed on the screen!

## Implementation
### Initial Stages
I realized that a similar problem has been defined before, which is standard facial expression classification. The dataset for this problem was available on Kaggle (linked below). The problem involves classifying images into the seven universal facial expressions:

- happiness
- surprise
- sadness
- fear
- disgust
- anger
- neutrality

I trained my model on this dataset.

### Preprocessing the Data
I noticed that there was a data imbalance in the given data. I tried to fix this by duplicating/flipping images, after which I was able to bring the model's accuracy from ~35% to ~45%. The accuracy was still low, but I figured this was due to the nature of the dataset, which had a lot of ambiguous images.

I also performed some other preprocessing steps such as converting the images to greyscale and resizing them. After comparing various classifiers offered by scikit-learn, I went with the random forest classifier due to its high accuracy and lowest error in the classification report.

### Training and Testing
I used the training and validation images to identify the model that performed best (Random Forest Classifier) and to identify hyperparameters, which again improved the accuracy to around 50%, after which I loaded the trained model into a .pkl file.

### Building the Demo Program
I used the trained model to process live video feed and capture emojis from it in real time. I used Mediapipe's inbuilt functionality for classifying hand gestures to classify gesture-related emojis.

### Dataset Used 
https://www.kaggle.com/datasets/msambare/fer2013

## Challenges
- Data imbalance (had to duplicate/flip images to resolve this)
- Issues displaying the image on the video (since OpenCV doesn't directly support emoji fonts, I had to overlay the emoji as an image onto the frame)
- Low accuracy due to use of traditional ML algorithms instead of neural networks (Although neural networks would have yielded better accuracy, I used scikit-learn due to my familiarity with the library and didn't explore other options until it was too late. I suppose the accuracy could be improved in the future by using neural networks instead.)
- Unavailability of adequate training data for other emojis (The data I found only pertained to the seven basic emotions and there was very little data available for training the model for other emojis, so I used Mediapipe to include some extra emojis.) 
