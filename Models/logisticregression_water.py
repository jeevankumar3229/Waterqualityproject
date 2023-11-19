import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")
data = pd.read_csv("./CSV/new_data.csv")
data.isnull().sum()
data.fillna(data.mean(),inplace=True)
data.isnull().sum()
data = np.array(data)
X = data[1:, :-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred=log_reg.predict_proba(X_test)
import warnings
warnings.filterwarnings('ignore')
# parameter grid
parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
model= GridSearchCV(log_reg,param_grid=parameters,scoring='accuracy',cv=100)
log_reg.get_params().keys()
y_pred=model.fit(X_train,y_train).predict_proba(X_test)
import joblib
joblib.dump(model, "Models\logisticregression_water.joblib")
