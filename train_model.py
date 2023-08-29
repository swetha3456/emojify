import os
import numpy as np
import cv2
import joblib

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train_path = "images/train"
test_path = "images/validation"
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

X_train = []
X_test = []
y_train = []
y_test = []

for label_id, label in enumerate(labels):
    label_train_path = os.path.join(train_path, label)
    label_test_path = os.path.join(test_path, label)
    
    for img_filename in os.listdir(label_train_path):
        img_path = os.path.join(label_train_path, img_filename)
        img = cv2.imread(img_path)  
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_train.append(gray_image.flatten())  
        y_train.append(label_id)
        
    for img_filename in os.listdir(label_test_path):
        img_path = os.path.join(label_test_path, img_filename)
        img = cv2.imread(img_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_test.append(gray_image.flatten())
        y_test.append(label_id)

print("Processed images")

X_train = np.array(X_train, dtype=object)
X_test = np.array(X_test, dtype=object)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Finished prepping data, beginning classification")

classifier = RandomForestClassifier(max_depth=30, min_samples_leaf=2, n_estimators=200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")

joblib.dump(classifier, 'model.pkl')