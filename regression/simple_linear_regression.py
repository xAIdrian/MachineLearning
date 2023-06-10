import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import datapreprocessing.data_preprocessing_tools as dpt

class SimpleLinearRegression:
  def __init__(self, dataset_filename, encoding_column_index):
      # Importing the dataset
      self.dataset = dpt.PreprocessingTools(dataset_filename)
      
      # Preparring the dataset for model
      self.X = self.apply_oneshot_encoding(encoding_column_index)
      self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split_dataset()
      
      # Training the model
      self.train_regressor()
      self.regressor_predictions()
      

  def apply_oneshot_encoding(self, column_index):
    self.X = self.dataset.encode_dependent_variable(column_index)  
    return self.X  
  
  def train_regressor(self):
    # training the simple linear regression model on the training set
    # regression is predicting continueous real values
    # classification is predicting categories or classes
    self.regressor = LinearRegression()
    self.regressor.fit(self.X_train, self.y_train)
    return self.regressor

  def regressor_predictions(self):  
    # predicting the test set results
    # predicted salaries from the num years experience
    self.y_train_pred = self.regressor.predict(self.X_train)
    self.y_pred = self.regressor.predict(self.X_test)
  
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



