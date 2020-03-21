import matplotlib.pyplot as plt
import numpy as np

#SAMPLE POINTS
X = [0 , 6 , 11 , 14 , 22 ]

Y = [1 , 7 , 12 , 15 , 21] 

#SOLVE FOR A AND B


def best_fit(X,Y):
    xbar=sum(X)/len(X)
    
    ybar=sum(Y)/len(Y)
    
    n=len(X) #or len(Y)
    
    
    numer= sum([ xi * yi for xi , yi in zip(X,Y) ]) - n *xbar * ybar
   
    
    denum=sum([xi**2 for xi in X]) -n *xbar ** 2
    
    
    
    b = numer / denum
    
    a=ybar - b * xbar
    
    
    print('best fit line :\ny={:.2f} + {:.2f}x'.format(a,b))
    
    print("xbar---> {} \n ybar---->{} \nnumer-->{} \ndenum --->{} \n b--->{} \na---->{}".format(xbar,ybar,numer,denum,b,a))
    return  a,b

#Solution
    
a , b = best_fit(X,Y)

#Best fit line :
#y=0.0 +0.92x


#plot points and fit line
plt.scatter(X,Y)

yfit=[a+b * xi for xi in X]
yfit
plt.plot(X,yfit)
plt.show()



from sklearn import datasets ,linear_model
from sklearn.metrics import mean_squared_error,r2_score


#Load the diabetes dataset

diabetes=datasets.load_diabetes()

#use only one feature

diabetes_X = diabetes.data[:,np.newaxis,2]
"""IMPORTANT NOTE : np.newaxis does change value between row and column such as : x=[4:1] ----->x=[1:4] """
# We're gonna working on body mass index



# Split the Data into training /testing sets

diabetes_X_train=diabetes_X[:-30]

diabetes_X_test=diabetes_X[-30:]

#Split the targets into trainig/testing sets

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]
#Target got things that measure of disease progression 




#Create linear regression object

regr=linear_model.LinearRegression()



#train the model using the training sets

regr.fit(diabetes_X_train, diabetes_y_train)


#Make predictions using the testing set


diabetes_y_pred=regr.predict(diabetes_X_test)


# The coefficients

print("Coefficietns :\n",regr.coef_ )

#The mean squarred error

print("Mean squared error: %.2f"
      %mean_squared_error(diabetes_y_test,diabetes_y_pred))


#explained variance score : 1 is perfect prediction


print("Variance score :%.2f" % r2_score(diabetes_y_test,diabetes_y_pred))


#Plot outputs

plt.scatter(diabetes_X_test , diabetes_y_test , color='yellow')
plt.plot(diabetes_X_test , diabetes_y_pred,color='blue' , linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()













