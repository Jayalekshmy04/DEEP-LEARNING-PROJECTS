## **Importing  libaries**

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## ***Loading data***

iris_data = pd.read_csv('/content/Iris (1).csv')
iris_data

# @id

from matplotlib import pyplot as plt
iris_data['Id'].plot(kind='hist', bins=20, title='id')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title PetalLengthCm

from matplotlib import pyplot as plt
iris_data['PetalLengthCm'].plot(kind='hist', bins=20, title='PetalLengthCm')
plt.gca().spines[['top', 'right',]].set_visible(False)

## ***Checking the data from Species cloumn***

iris_data['Species'].unique()

## ***Splitting the data into features and label***

# Split the dataset into features and labels
X = iris_data.drop(['Species','Id'] ,axis=1)
y = iris_data['Species']

## ***Splitting the data into  training and testing data***

training data = 80% of data                                                       
testing data = 20% of data


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## ***Initialize individual classifiers***

# Initialize individual classifiers
logreg = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', probability=True, random_state=42)   #In the provided ensemble code, probability=True is set for the SVM (SVC) because it's part of the ensemble, and the ensemble uses soft voting
naive_bayes = GaussianNB()
random_forest = RandomForestClassifier(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)


## ***Create an ensemble using a soft voting approach***

# Create an ensemble using a soft voting approach (based on class probabilities)
ensemble = VotingClassifier(estimators=[
    ('logreg', logreg),
    ('knn', knn),
    ('svm', svm),
    ('naive_bayes', naive_bayes),
    ('random_forest', random_forest),
    ('xgb_classifier', xgb_classifier)
], voting='soft')

key : to find probability use hard & to find vote use soft

Types of voting


## **how voting = soft works**

**Classifier A:**

Probability of Class 0: 0.6

Probability of Class 1: 0.4

**Classifier B:**

Probability of Class 0: 0.3

Probability of Class 1: 0.7

Soft voting involves summing up these probabilities for each class and choosing the class with the highest cumulative probability.

Total probabilities for Class 0:
0.6
+
0.3
=
0.9



Total probabilities for Class 1:
0.4
+
0.7
=
1.1



Since Class 1 has a higher cumulative probability (1.1), the soft voting approach would predict Class 1 as the final output.

## **how voting = hard works**

consider three classifiers, A, B, and C, predicting whether an example belongs to Class 0 or Class 1.

**Classifier A predicts Class 0.**

**Classifier B predicts Class 1.**

**Classifier C predicts Class 1.**

In hard voting, we count the votes for each class:

Votes for Class 0: 1 (from Classifier A)

Votes for Class 1: 2 (from Classifiers B and C)

Since Class 1 has more votes, the hard voting approach would predict Class 1 as the final output

# Fit the ensemble model on the ensemble.fit(X_train, y_train)training data
ensemble.fit(X_train, y_train)

# Make predictions on the test set
predictions = ensemble.predict(X_test)

# Evaluate the ensemble's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Ensemble Accuracy: {accuracy}")

input_data=(5.1,3.5,1.4,0.2)
input_data_as_numpy_array=np.array(input_data)
reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=ensemble.predict(reshaped)
print(f"the species of iris flower is identified as {prediction[0]}")