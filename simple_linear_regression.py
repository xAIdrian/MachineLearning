import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datapreprocessing.data_preprocessing_tools as dpt

if __name__ == "__main__":
  # Importing the dataset
  dataset = dpt.PreprocessingTools('Salary_Data.csv')
  X, y = dataset.import_dataset()
  X_train, X_test, y_train, y_test = dataset.split_dataset()

  print(X)
  print('----------------')
  print(y)
  print('----------------')
  print(X_train)
  print('----------------')
  print(X_test)
  print('----------------')
  print(y_train)
  print('----------------')
  print(y_test)

