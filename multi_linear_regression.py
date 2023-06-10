import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import regression.simple_linear_regression as slr

if __name__ == "__main__":
    np.set_printoptions(precision=2)

    # Importing the dataset
    # Encode the categorical data
    # Splitting the dataset into the Training set and Test set
    # Training the Multiple Linear Regression model on the Training set
    slr = slr.SimpleLinearRegression('50_Startups.csv', 3)

    print(np.concatenate((slr.y_pred.reshape(len(slr.y_pred), 1), slr.y_test.reshape(len(slr.y_test), 1)), axis=1))
    slr.plot_simple_linear_regression_test_set(
        title='Startups profits',
        x_label='Startups',
        y_label='Profits'
    )
    
    # Predicting the Test set results
