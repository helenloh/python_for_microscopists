import time

import tensorflow as tf
import keras
# TensorFlow and Keras for deep learning

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout, Flatten
# Importing various layers and models from Keras

import numpy as np
from matplotlib import pyplot as plt
# Numpy for numerical operations and pyplot for plotting

from sklearn.model_selection import GridSearchCV
# GridSearchCV for optimizing hyperparameters

print(tf.__version__)
print(keras.__version__)
# Printing TensorFlow and Keras versions for reference

np.random.seed(42)
# Setting a seed for reproducibility

# Loading the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Plotting the first 20 images in the dataset
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])

# Normalizing the image data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Splitting training data for grid search
from sklearn.model_selection import train_test_split
x_grid, x_not_use, y_grid, y_not_use = train_test_split(x_train, y_train, test_size=0.98, random_state=42)

# Reshaping data for the convolutional network
x_grid = np.expand_dims(x_grid, axis=3)

# Define the size based on reshaped dataset
SIZE = x_grid.shape[1]

# Defining a function for the convolutional feature extractor
def feature_extractor():       
    activation = 'sigmoid'
    feature = Sequential()
    
    feature.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 1)))
    feature.add(BatchNormalization())
    
    feature.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
    feature.add(BatchNormalization())
    feature.add(MaxPooling2D())
    
    feature.add(Flatten())
    
    return feature

# Creating the feature extractor and printing its summary
feature_extractor = feature_extractor()
print(feature_extractor.summary())


# Starting the training time measurement
start_time = time.time()


# Extracting features for training the Random Forest
X_for_RF = feature_extractor.predict(x_grid)

# Importing and training Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(X_for_RF, y_grid) # Training with feature extracted data

# Extracting features from test data
X_test_feature = feature_extractor.predict(np.expand_dims(x_test, axis=3))

# Predicting using the trained Random Forest model
prediction_RF = RF_model.predict(X_test_feature)

# Evaluating the Random Forest model
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))

# Generating and printing a confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
sns.heatmap(cm, annot=True)

# Setting up parameters for GridSearchCV
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1], 
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [10,20,30]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1]
        }
    }
}

# Performing Grid Search to find the best model 
# and hyperparameters

#in this line we use three model to do model fitting
#and finding the best model
# three models (SVM, Random Forest,
# and Logistic Regression) within the grid search
scores = []

for model_name, mp in model_params.items():
    grid =  GridSearchCV(estimator=mp['model'], 
                         param_grid=mp['params'], 
                         cv=5, n_jobs=16, 
                         return_train_score=False)
    
    grid.fit(X_for_RF, y_grid)
    
    scores.append({
        'model': model_name,
        'best_score': grid.best_score_,
        'best_params': grid.best_params_
    })

# Ending the training time measurement
end_time = time.time()

# Calculating and printing the total training time
total_time = end_time - start_time
print(f"Training Time: {total_time} seconds")


# Creating a DataFrame to display the results of Grid Search
import pandas as pd    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print('dataframe')
print(df)


