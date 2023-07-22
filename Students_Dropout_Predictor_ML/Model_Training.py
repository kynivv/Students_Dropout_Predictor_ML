import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score as asc
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


# Import Data
df = pd.read_csv('students_info.csv')


# EDA
print(df.isnull().sum())
print('-' * 100)

print(df.dtypes)
print('-' * 100)

print(df['Target'].value_counts())
print('-' * 100)


# Data Preproccesing
df.drop(df[df['Target'] == 'Enrolled'].index, inplace = True)
print (df['Target'].value_counts())
print (df.shape)
print('-' * 100)

le = LabelEncoder()

for t in df.columns:
    if df[t].dtype == object:
        df[t] = le.fit_transform(df[t])

print(df['Target'])
print('-' * 100)


# Train Test Split
target = df['Target']
features = df.drop('Target', axis=1)

print(target)
print('-' * 100)

print(features)
print('-' * 100)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.20, random_state=42)


# Model Training
models = [ExtraTreeClassifier(), RidgeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), DummyClassifier()]

for m in models:
    print(m)
    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train accuracy is : {asc(Y_train, pred_train)}')
    
    pred_test = m.predict(X_test)
    print(f'Test accuracy is : {asc(Y_test, pred_test)}')

    print('=' * 100)