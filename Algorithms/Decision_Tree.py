#Starting implementation
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#matplotlib inline

import numpy as np

import seaborn as sns

from sklearn import tree

df = pd.read_csv("iris_df.csv")

df.columns = ["X1" , "X2" , "X3" , "X4" , "Y"]

df.head()


#implementation

from sklearn.model_selection import train_test_split

decision = tree.DecisionTreeClassifier(criterion = "gini")

X = df.values[:,0:4]
#  X got that got column named X1,X2....

Y = df.values[:,4] 
# you can consider like result

trainX ,testX , trainY ,testY = train_test_split(X,Y,test_size=0.3) 
#now , there'r different five column such as X1 , X2 , X3 , X4 ,Y.
#we splited up data into the X,we also applied the same processing for Y(%30).
# trainX got %60 of X and trainY got %60 of Y , and others got %30 of obtain data.





decision.fit(trainX , trainY)

print("Accuracy:\n",decision.score(testX , testY))

#Visualisation

from sklearn.externals.six import StringIO

from IPython.display import Image

import pydotplus as pydot

dot_data=StringIO()

tree.export_graphviz(decision , out_file=dot_data)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())


























