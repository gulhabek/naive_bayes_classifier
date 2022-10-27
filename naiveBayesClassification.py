# Making imports
import numpy as np
import pandas as pd

# load data set
data=pd.read_csv("C:\\Users\\toshiba\\Desktop\\vm\\lab\\4.HAFTA_NAİVE_BAYES\\SosyalMedyaReklamKampanyasi.csv")

# Separating the Data Set into Dependent and Independent Attributes
X=data.iloc[:,[2,3]]
y=data.iloc[:,4]

# Separating Data as Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating and Training a Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
cl=GaussianNB()
cl.fit(X_train,y_train)

# Making prediction
y_pred=cl.predict(X_test)

# confusion_matrix
import sklearn.metrics as mt
cm = mt.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = mt.confusion_matrix(y_test, y_pred).ravel()

# metrics
#print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn))
print("Accuracy: ", mt.accuracy_score(y_test,y_pred))

#print("Precision Score:  ", tp/(tp+fp))
print("Precision Score: ", mt.precision_score(y_test, y_pred))

#print("Recall Score: ", tp/(tp+fn))
print("Recall: ",mt.recall_score(y_test, y_pred))

# p=tp/(tp+fp)
# r=tp/(tp+fn)

#print("F1 ölçütü: ",(2*p*r)/(p+r))
print("F1 ölçütü: ",mt.f1_score(y_test, y_pred))
