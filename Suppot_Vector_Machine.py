from sklearn import svm
import seaborn as sns
import pandas as pd
df=pd.read_csv('iris_df.csv')

df.columns = ['X4' , 'X3' , 'X1' ,'X2' , 'Y']

df = df.drop(['X4' , 'X3' ] , 1)

df.head()

from sklearn.model_selection import train_test_split

support= svm.SVC()

X=df.values[:,0:2]
Y=df.values[:,2]

trainX ,testX , trainY ,testY  = train_test_split(X , Y ,test_size=0.3)

sns.set_context('notebook' , font_scale=1.1)

sns.set_style('ticks')

sns.lmplot('X1' , 'X2' ,scatter=True , fit_reg = False ,  data = df , hue = 'Y')

plt.ylabel('X2')

plt.xlabel('X1')