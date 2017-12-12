# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:15:34 2017
@author: rajesh.unnikrishnan
"""

# Introduction to Machine Learning
# Use the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def cancer_predict():
    
    #The object returned by `load_breast_cancer()` is a scikit-learn Bunch object, which is similar to a dictionary
    cancer = load_breast_cancer()
    # Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame. 
    # Convert the sklearn.dataset `cancer` to a DataFrame. 
    cancer_df=pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'].tolist())
    cancer_df['target']=cancer['target']
    print('Shape of Cancer DataFrame:',cancer_df.shape,'\n')
    # What is the class distribution? (i.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1)?)
    target=cancer_df[['target','mean radius']].groupby(['target']).count()
    target.index=['malignant', 'benign']
    
    print('Class distribution in Cancer Dataset:','\n',target.iloc[:,0],'\n')
    
    X=cancer_df.drop('target',axis=1)
    y=cancer_df['target']
    
    # Using `train_test_split`, split `X` and `y` into training and test sets `(X_train, X_test, y_train, and y_test)`.
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    # Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with `X_train`, `y_train` and using one nearest neighbor (`n_neighbors = 1`).
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train,y_train)
    # Using your knn classifier, predict the class label using the mean value for each feature.
    # Hint: You can use `cancerdf.mean()[:-1].values.reshape(1, -1)` which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).
    # *This function should return a numpy array either `array([ 0.])` or `array([ 1.])`*
    # means = cancer_df.mean()[:-1].values.reshape(1, -1)
    # print ('predict',knn.predict(means))
    # Using your knn classifier, predict the class labels for the test set `X_test`.
    
    print('Predicted class labels for the test set:','\n',knn.predict(X_test),'\n')
    # Find the score (mean accuracy) of your knn classifier using `X_test` and `y_test`.
    print('Score of knn Classifier: ',knn.score(X_test,y_test),'\n')
   
    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    
cancer_predict()


