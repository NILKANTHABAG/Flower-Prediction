# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:10:53 2021

@author: NILKANTHA BAG
"""

import pandas as pd
import numpy as np
import pickle


iris = pd.read_csv('Iris.csv')

iris.info()
iris.describe()
iris.apply(np.max)
iris.corr()


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth = 3)
classifier.fit(X_train, y_train)
print("Training completed")

y_test_pred = classifier.predict(X_test)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)

con_metric = metrics.confusion_matrix(y_test, y_test_pred)
con_metric


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(25,10))
f = plot_tree(classifier, feature_names = X_train.columns, 
              class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 
              filled = True, 
              rounded = True, 
              fontsize = 14
             )
pickle.dump(classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0.8,0.3,0.1,0.5]]))





