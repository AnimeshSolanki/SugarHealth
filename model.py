import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

def train_model():
    diabetes_dataset = pd.read_csv('diabetes.csv')

    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    X = standardized_data

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    return classifier, scaler

def predict_diabetes(classifier, scaler, input_data):
    input_np_array = np.asarray(input_data)
    input_reshaped = input_np_array.reshape(1, -1)

    std_data = scaler.transform(input_reshaped)

    prediction = classifier.predict(std_data)

    return "Positive" if prediction[0] == 1 else "Negative"