import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # importing datasets
  # dependent variable is last column: purchased or not
  # features are all columns except last one. country, age, salary
  users_dataset = pd.read_csv('users.csv')  # creates dataframe
  X = users_dataset.iloc[: , :-1].values # select all the rows : && all the columns except last one. all columns excluding the last one :-1
  y = users_dataset.iloc[: , -1].values

  print(X)
  print(y)
