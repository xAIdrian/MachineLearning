import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
  # 1. Importing datasets
  # dependent variable is last column: purchased or not
  users_dataset = pd.read_csv('users.csv')  # creates dataframe
  X = users_dataset.iloc[: , :-1].values # select all the rows : && all the columns except last one. all columns excluding the last one :-1
  y = users_dataset.iloc[: , -1].values

  # 2. Taking care of missing data
  imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

  # fit method will find the missing values and replace them with the mean of the column
  # upper bound is excluded
  imputer.fit(X[:, 1:3]) 

  # transform method will replace the missing values with the mean of the column
  X[:, 1:3] = imputer.transform(X[:, 1:3]) 

  # 3. Encoding categorical data
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
  X = np.array(ct.fit_transform(X))
  
  # 4. Encoding the dependent variable
  le = LabelEncoder()
  y = le.fit_transform(y)

  # 5. Splitting the dataset into the Training set and Test set
  # apply feature scaling after splitting the dataset to avoid information leakage from test set to training set
  # random_state=1 to get the same results consistently, for learning
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

  print('Training and Test Data')
  print(X_train)
  print('----------------')
  print(X_test)
  print('----------------')
  print(y_train)
  print('----------------')
  print(y_test)

  # 6. Feature Scaling
  # Feature scaling prevents some features from dominating others. Features == columns
  # Not used all of the time
  sc = StandardScaler()

  # you do not need to apply feature scaling to dummy variables. all the dummy variables are already in the same scale
  # take the 3rd column to the end
  X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
  X_test[:, 3:] = sc.transform(X_test[:, 3:])

  print('Feature Scaling Data')
  print(X_train)
  print('----------------')
  print(X_test)
