import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import warnings
import pickle
warnings.filterwarnings("ignore")
data = pd.read_csv("./CSV/new_data.csv")

data.isnull().sum()
data.fillna(data.mean(),inplace=True)
data.isnull().sum()
X = data.drop('Potability',axis=1)
Y = data['Potability']
X_train , X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2, shuffle=True, random_state=101)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
tr=ExtraTreesClassifier()
tr.fit(X_train,Y_train)
y_pred=tr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_pred,Y_test)*100)
parameter={
    'n_estimators':[100,150,200,250],
    'criterion':['gini','entropy'],
    'min_samples_split':[2,10],
    'min_samples_leaf':[2,10]
}
model=GridSearchCV(tr,param_grid=parameter,scoring='accuracy',cv=20)
y_pred=model.fit(X_train,Y_train).predict(X_test)
print(accuracy_score(y_pred,Y_test)*100)
import joblib
joblib.dump(model, "Models\extratree_water.joblib")