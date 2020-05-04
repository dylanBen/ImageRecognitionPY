
# Import Global & Machine Learning Libraries
import pandas as pd
import numpy as np
import time
start_time = time.time()

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

test: ImageDataGenerator

test_generator = ImageDataGenerator(rescale=1/255.0)
test = test_generator.flow_from_directory(
    directory='./data/test/',
    target_size=(250, 250),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=312,
)

print("process test data..")
test_normal = test[0]
test_pneumonia = test[1]

test_normal_data = test_normal[0].reshape(len(test_normal[1]), -1)
test_normal_target = test_normal[1]
test_pneumonia_data = test_pneumonia[0].reshape(len(test_pneumonia[1]), -1)
test_pneumonia_target = test_pneumonia[1]

test_data = [*test_normal_data, *test_pneumonia_data]
test_target = [*test_normal_target, *test_pneumonia_target]

# load the model from disk
model = joblib.load('./Logistic_Regression/Logistic_Regression.sav')

# Model Prediction
print("model predict..")
pred_target = model.predict(test_data)

# Compare Value in a Dataframe
comp=pd.DataFrame(pred_target, test_target)

print("La précision du modèle en %:", 100 * accuracy_score(test_target, pred_target))

print("confusion matrix:")
print(confusion_matrix(test_target, pred_target))

print("Finish! total time:", time.time() - start_time)
