from sklearn.cluster import KMeans 
import pandas as pd
from sklearn.model_selection import train_test_split 
from matplotlib import style 
from sklearn.cluster import KMeans 
import seaborn as sns 


df = pd.read_csv('iris_df.csv') 
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y'] 


df = df.drop(['X4', 'X3'], 1) 
df.head() 

kmeans = KMeans(n_clusters=3) 
X = df.values[:, 0:2]
 
kmeans.fit(X) 
df['Pred'] = kmeans.predict(X) 
df.head() 
sns.set_context('notebook', font_scale=1.1) 
sns.set_style('ticks') 
sns.lmplot('X1','X2', scatter=True, fit_reg=False, data=df, hue = 'Pred') 


### For the purposes of this example, we store feature data from our 
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into 
### a feature matrix `X` before entering it into the algorithm.
f1 = df['Distance_Feature'].values

f2 = df['Speeding_Feature'].values

X=np.matrix(zip(f1,f2))
kmeans = KMeans(n_clusters=2).fit(X)