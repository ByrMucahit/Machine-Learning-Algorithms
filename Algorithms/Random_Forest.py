from sklearn.ensemble import RandomForestClassifier 

import pandas as pd

df = pd.read_csv('iris_df.csv') 
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y'] 
df.head() 
from sklearn.model_selection import train_test_split 
forest = RandomForestClassifier() 
X = df.values[:, 0:4] 
Y = df.values[:, 4] 
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3) 
forest.fit(trainX, trainY) 
print('Accuracy: \n', forest.score(testX, testY)) 
pred = forest.predict(testX)

from sklearn.ensemble import RandomForestClassifier 
X = [[0, 0], [1, 1]] 
Y = [0, 1] 
clf = RandomForestClassifier(n_estimators=10) 
clf = clf.fit(X, Y)



from sklearn import decomposition 
df = pd.read_csv('iris_df.csv') 
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y'] 
df.head() 

pca = decomposition.PCA() 
fa = decomposition.FactorAnalysis() 
X = df.values[:, 0:4] 
Y = df.values[:, 4] 
train, test = train_test_split(X,test_size = 0.3) 
train_reduced = pca.fit_transform(train) 
test_reduced = pca.transform(test) 
pca.n_components_ 


