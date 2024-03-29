import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


#this's a the test set ,it's straight line with some Gaussian Noise


xmin , xmax = -10 , 10

n_samples = 100

np.random.seed(0)

X=np.random.normal(size = n_samples)

y=(X > 0).astype(np.float)

X[X>0]*=4


X += .3 * np.random.normal(size = n_samples)

X= X[:,np.newaxis]

# Run the classifier

clf = linear_model.LogisticRegression(C=1e5)

clf.fit(X,y)

#and plot result


plt.figure(1 , figsize=(4,3))

plt.clf()

plt.scatter(X.ravel() , y , color ='black' , zorder = 20)

X_test = np.linspace(- 10 , 10 , 300)


def model(x):
    return 1 / (1+np.exp(-x))

loss = model(X_test * clf.coef_ + clf.intercept_).ravel()

plt.plot(X_test , loss , color ='blue' , linewidth=3)

ols = linear_model.LinearRegression()

ols.fit(X,y)

plt.plot(X_test , ols.coef_*X_test + ols.intercept_ , linewidth=1)

plt.axhline(.5 , color='.5') 

plt.ylabel('y')
plt.xlabel('X')

plt.xticks(range(-10 , 10))
plt.yticks([0,0.5,1])

plt.ylim(-.25 , 1.25)

plt.xlim(-4 , 10)

plt.legend(('Logistic Regression Model' , 'Linear Regression Model'),
           loc="lower right" , fontsize='small')
plt.show() 


