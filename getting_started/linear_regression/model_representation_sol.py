import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(w, b, x):
    """
    Computes the prediction of a linear model
    Args:
        x (ndarray (m, 1)): Data, m examples
        w, b (scalar): model parameters
    Returns:  
        y ndarray (m,): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

if __name__ == "__main__":
  # plt.style.use('./deeplearning.mplstyle')

  # x_train is the input variable (size in 1000 square feet)
  # y_train is the output variable (price in 1000 dollars)
  x_train = np.array([1.0, 2.0])
  y_train = np.array([300.0, 500.0])

  print('x_train: ', x_train)
  print('y_train: ', y_train)

  # we'll use m to decode the # of training examples
  print(f"x_train_.shape: {x_train.shape}")
  m = x_train.shape[0]
  m_len = len(x_train)
  print(f"the number of training examples: {m}")

  # get theh specific ith features
  i = 1
  print(f"x_train[{i}]: {x_train[i]}")
  print(f"y_train[{i}]: {y_train[i]}")

  # plot the training data
  # plt.scatter(x_train, y_train, marker='x', c='r')
  # plt.title('Housing Prices')
  # plt.xlabel('Size in 1000s square feet')
  # plt.ylabel('Price in 1000s dollars')
  # plt.show()

  # let's compute the value of linear regression
  w = 200
  b = 100
  print(f"w: {w}")
  print(f"b: {b}")

  # compute the prediction of the model
  tmp_f_wb = compute_model_output(w, b, x_train)
  
  # plot
  plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
  plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

  plt.title('Housing Prices')
  plt.xlabel('Size in 1000s square feet')
  plt.ylabel('Price in 1000s dollars')
  plt.legend()
  plt.show()
