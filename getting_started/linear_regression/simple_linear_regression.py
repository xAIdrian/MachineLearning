import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import datapreprocessing.data_preprocessing_tools as dpt

class SimpleLinearRegression:
  def __init__(self, dataset_filename, encoding_column_index=None, start_column_index=None, split_dataset=True):
      # Importing the dataset
      if (start_column_index is not None):
        self.dataset = dpt.PreprocessingTools(dataset_filename, start_column_index)
      else:
        self.dataset = dpt.PreprocessingTools(dataset_filename)
      
      # Preparring the dataset for model
      if (encoding_column_index is not None):
        self.X = self.apply_oneshot_encoding(encoding_column_index)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split_dataset()
      
      # Training the model
      self.train_regressor(split_dataset)
      self.regressor_predictions(split_dataset)
      

  def apply_oneshot_encoding(self, column_index):
    self.X = self.dataset.encode_dependent_variable(column_index)  
    return self.X  
  
  def train_regressor(self, split_dataset):
    # training the simple linear regression model on the training set
    # regression is predicting continueous real values
    # classification is predicting categories or classes
    self.regressor = LinearRegression()
    if (split_dataset):
      self.regressor.fit(self.X_train, self.y_train)
    else:
      self.regressor.fit(self.dataset.X, self.dataset.y)
    return self.regressor

  def regressor_predictions(self, split_dataset):  
    # predicting the test set results
    if (split_dataset):
      self.y_train_pred = self.regressor.predict(self.X_train)
      self.y_pred = self.regressor.predict(self.X_test)
    else:
      self.y_pred = self.regressor.predict(self.X)
  
  def plot_simple_linear_regression(
    self,    
    X_dependents, 
    y_independent, 
    y_prediction,
    title,
    x_label,
    y_label,
    X_train_dependents=None,
  ):
    if X_train_dependents is None:
      X_train_dependents = X_dependents

    plt.scatter(X_dependents, y_independent, color='red')
    plt.plot(X_train_dependents, y_prediction, color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

  def plot_simple_linear_regression_training_set(self, title, x_label, y_label):
    if (self.y_train_pred is None):
      raise ValueError('y_train_pred is None. Please train the model first.')

    self.plot_simple_linear_regression(
      self.X_train,
      self.y_train,
      self.y_train_pred,
      title,
      x_label,
      y_label,
    )

  def plot_simple_linear_regression_test_set(self, title, x_label, y_label):
    self.plot_simple_linear_regression(
      self.X_test,
      self.y_test,
      self.y_pred,
      title, 
      x_label,
      y_label,
      self.X_train
    )  

# if __name__ == "__main__":
#   simple_linear_regression = SimpleLinearRegression('Salary_Data.csv')
#   # can only show one or the other.
#   simple_linear_regression.plot_simple_linear_regression_training_set(
#     'Salary vs Experience (Training Set)',
#     'Years of Experience',
#     'Salary'
#   )
#   simple_linear_regression.plot_simple_linear_regression_test_set(
#     'Salary vs Experience (Test Set)',
#     'Years of Experience',
#     'Salary',
#   )



