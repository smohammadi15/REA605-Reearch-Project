#!/usr/bin/python3
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import glob,os
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


csvlist = "/home/user/Downloads/Flowmeter_CSVs/*.csv"
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', csvlist))))
df.to_csv('./Flowmeter_CSVs/compiled.csv',index=False)

csv='./Flowmeter_CSVs/compiled.csv'
dataset=pd.read_csv(csv,index_col=False)

#dataset = pd.concat(pd.read_csv("cleaned.csv",index_col=False))
#print(dataset.head())
#print(dataset.shape)
#dataset.plot(kind='box',subplots=True,layout=(1,2),sharex=False,sharey=False)
#dataset.hist()
#matplotlib.show()

array = dataset.values
X=array[:,0:57]
Y=array[:,57]
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size,random_state=seed)

#=dataset.iloc[:, :-1].values
#Y=dataset.iloc[:, 56].values

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size=0.2,random_state=0)
#print(dataset.shape)

#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)



scoring="accuracy"
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='auto')))
#models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]
for name,model in models:
	kfold=model_selection.KFold(n_splits=10,random_state=0)
	cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg="%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
	print(msg)


print('======')
print('KNN')
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print('======')
print('CART')
# Make predictions on validation dataset
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print('======')
print('SVM')
# Make predictions on validation dataset
svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

