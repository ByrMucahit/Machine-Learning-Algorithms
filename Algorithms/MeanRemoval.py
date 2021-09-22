# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:25:54 2020

@author: mücahit
"""

import numpy as np
from sklearn import preprocessing

#                                   STANDARDİZED 
#We need to convert values into data to  between one and zero  to get more succesfully result  

# We prepared data to operate.
input_data=np.array([[3,-1.5,3,-6.4],[0,3,-1.3,4.1],[1,2.3,-2.9 ,-4.3]]) 

#We scale out to data
data_Standardized=preprocessing.scale(input_data) 

print("\n Mean=",data_Standardized.mean(axis=0))

print ("Std deviation =",data_Standardized.std(axis=0))


#                                   SCALİNG
# We decided to which range change to this value in data ...
Data_Scaler=preprocessing.MinMaxScaler(feature_range=(0,1))

# we applied
data_scaled=Data_Scaler.fit_transform(input_data)

# scaling's formul

#   ( ANY NUMBER - MEAN VALUES ) / STANDARD DEVİATİON


print("\nMin max scaled =",data_scaled)





#                   NORMALİZATİON
Data_Normalized=preprocessing.normalize(input_data,norm='l1')

# Normalization's formule =  (ANY NUMBER - MİN(ANY NUMBER))/[MAX(X) - MİN(X)]
print("\nNormalied Data=",Data_Normalized)




#                   BİNARİZATİON

# here used to convert value of array to binary value accoringto threshold value
Data_Binarized=preprocessing.Binarizer(threshold=1.4).transform(input_data)

print("\nBinarized Data:",Data_Binarized)




#               ONE-HOT-ENCODER

Encoder=preprocessing.OneHotEncoder()
Encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
encoded_vector = Encoder.transform([[2,3,5,3]]).toarray()
print("\nEncoderd Vector--->",encoded_vector)



#           LABEL ENCODİNG

Etiket_Kodlayıcı = preprocessing.LabelEncoder()

Araba_Markaları = ['suzuki' , 'ford' , 'suzuki' , 'suzuki' , 'toyota' , 'ford' , 'bmw' ]

Etiket_Kodlayıcı.fit(Araba_Markaları)

print("\n Class mapping:")
    
for i , item in enumerate(Etiket_Kodlayıcı.classes_):
    print(item,"-->",i)
    
    
labels=['toyota' , 'ford','suzuki']

encoded_labels =Etiket_Kodlayıcı.transform(labels)
print("\nLabels:",labels)

print("Encoded Labels:",list(encoded_labels))




#                        DATA ANALYSIS

import pandas 

data = 'pima-indians-diabetes.csv'

names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']

dataset=pandas.read_csv(data,names=names)

# DIMENSIONS OF DATASET

print(dataset.shape)

#LIST THE ENTIRE DATA

print(dataset.head(20))

#View the Statistical Summary

print(dataset.describe())


#               UNİVARİATE PLOTS

# which has a variable

import matplotlib.pyplot as plt

data2 = 'iris_df.csv'

names= ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'class']

dataset2=pandas.read_csv(data2 , names = names)

dataset2.plot(kind='box' , subplots=True , layout=(2,2) , sharex = False , sharey = False)
plt.show()

#BOX AND WHİSKER PLOTS

#Histograms

dataset.hist()

#plt().show()



#MULTİVARİATE PLOTS(whic has varaiblen more then 1)

from pandas.plotting import scatter_matrix


scatter_matrix(dataset2)

plt.show()



