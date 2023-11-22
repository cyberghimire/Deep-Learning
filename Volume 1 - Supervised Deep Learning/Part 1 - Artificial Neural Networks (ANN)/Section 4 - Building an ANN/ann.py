#Part 1: Data Preprocessing


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2: Making the ANN
# Importing the Keras libraries and packages
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units= 6, activation = 'relu'))
classifier.add(Dropout(rate=0.1))

#Adding the final layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3: Making predictions to evaluate the model

#Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred>0.5

#Predicting a single new observation
"""
Predict if a customer with the following informations will leave the bank:
    Geography: France
    Credit score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance: 60000
    Number of products: 2
    Has Credit Card: Yes
    Is Active Member: Yes
    Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000 ]])))
new_prediction = new_prediction>0.5

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Part 4 - Evaluating, Improving and Tuning the ANN

#Evaluating the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units= 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier 

classifier = KerasClassifier(build_fn = build_classifier,  batch_size=10, epochs=40 )

accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()
        
    
# Improving the ANN
#Dropout Regularizatoin to reduce overfitting if needed

# Tuning the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units= 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier 

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [40, 50],
              'optimizer': ['adam', 'rmsprop']
              }

grid_search = GridSearchCV(estimator= classifier, param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_








