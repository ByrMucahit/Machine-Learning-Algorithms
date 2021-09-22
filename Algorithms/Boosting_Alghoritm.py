#for classification 
from sklearn.ensemble import AdaBoostClassifier  
#for regression 
from sklearn.ensemble import AdaBoostRegressor  
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier()  
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1) 
#Here we have used decision tree as a base estimator; Any ML learner can be used as base #estimator if it accepts sample weight  
clf.fit(x_train,y_train) 
