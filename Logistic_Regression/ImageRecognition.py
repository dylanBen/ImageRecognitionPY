
# Import Global & Machine Learning Libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

import time
start_time = time.time()
print("start")

train: ImageDataGenerator
test: ImageDataGenerator

train_generator = ImageDataGenerator(rescale=1/255.0)
train = train_generator.flow_from_directory(
    directory='./data/train/',
    target_size=(250, 250),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=326,
)

test_generator = ImageDataGenerator(rescale=1/255.0)
test = test_generator.flow_from_directory(
    directory='./data/test/',
    target_size=(250, 250),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=312,
)

print("process train data")
train_normal = train[0]
train_pneumonia = train[1]

# for reshape : len(train[0][1]) == batch_size
train_normal_data = train_normal[0].reshape(len(train_normal[1]), -1) 
train_normal_target = train_normal[1]
train_pneumonia_data = train_pneumonia[0].reshape(len(train_pneumonia[1]), -1)
train_pneumonia_target = train_pneumonia[1]

# train_pneumonia_data.shape

train_data = [*train_normal_data, *train_pneumonia_data]
train_target = [*train_normal_target, *train_pneumonia_target]

print("process test data")
test_normal = test[0]
test_pneumonia = test[1]

test_normal_data = test_normal[0].reshape(len(test_normal[1]), -1)
test_normal_target = test_normal[1]
test_pneumonia_data = test_pneumonia[0].reshape(len(test_pneumonia[1]), -1)
test_pneumonia_target = test_pneumonia[1]

# test_pneumonia_data.shape

test_data = [*test_normal_data, *test_pneumonia_data]
test_target = [*test_normal_target, *test_pneumonia_target]

# Model Creation
model=LogisticRegression(max_iter=6000)

# Model Training
print("model training")
model.fit(train_data,train_target)

# Predict y_pred
print("model predict")
pred_target = model.predict(test_data)

# Compare Value in a Dataframe
comp=pd.DataFrame(pred_target, test_target)

print("La précision du modèle:", accuracy_score(test_target, pred_target))


print("confusion matrix")
print("", confusion_matrix(test_target, pred_target))

print("Finish! total time", time.time() - start_time)
