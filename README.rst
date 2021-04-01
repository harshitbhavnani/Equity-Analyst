***************
Equity Analyst
***************

Equity Analyst is an Open-Source Python Package that uses a Deep Learning model to predict future stock prices based on historical values. Additionally, it also acts as an aid to acquire and visualise stock prices without any redundancy.

**Installing the Package**

    ``$ pip3 install equityanalyst``

Function Description
""""""""""""""""""""

1. ``y_pred = stock_price_predict(company)``
********************************************

**The function performs the following tasks:**

- Accepts the ticker name of the stock and scrapes historic data of the stock using the Yahoo Finance API. The start date and end date for the historic data are 2000 days and 60 days ago respectively.

- A Recurrent Neural Network model is trained with the data. This model is used to make predictions on the data for a number of days that need to be taken into consideration. This parameter is defined by the user.

- Plots a graph that displays our predicted values along with the actual values. This will act as a visual aid for the users to comprehend the accuracy of the model on the particular stock.

- Returns predicted values for the desired number of days which is defined by the user.

2. ``df = get_stock_data(company)``
***********************************

This function acquires data using the Yahoo Finance API for a particular duration which is specified by the user.

3. ``plot_stock_data(company)``
*******************************

This function visualises the stock prices for a particular duration specified by the user.

**Importing all Functions from the package**

    ``from equityanalyst import stock_price_predict``

    ``from equityanalyst import get_stock_data``

    ``from equityanalyst import plot_stock_data``


Variable Description:
"""""""""""""""""""""
company(string): Ticker name of the stock

prediction_days(int): Number of days that need to be taken into consideration for predictions

days_pred(int): Number of days that we need projections for

y_pred(array): Predictions for 'days_pred' number of days

Output
""""""

.. image:: https://github.com/harshitbhavnani/Stock-Price-Predictor/blob/main/Screenshot%202021-03-27%20021305.png?raw=true
           :alt: Output
           :height: 350px
           :width: 518px

Dependencies
""""""""""""

The libraries that need to be installed for the package to work are as follows:

- Matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

    ``$ pip3 install matplotlib``

- Tensorflow: TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow

    ``$ pip3 install tensorflow``

- Numpy: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

    ``$ pip3 install numpy``

- Pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

    ``$ pip3 install pandas``

- Keras: Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research.

    ``$pip3 install keras``

- Pandas Data Reader: Pandas Data Reader is used to extract historic data of stocks by calling the Yahoo API

    ``$pip3 install pandas-datareader``

- TQDM: This package instantly make your loops show a smart progress meter for the fitting of the RNN model

    ``$pip3 install tqdm``


- Sklearn: MinMaxScaler function from Sklearn is used in order to standardise the closing values between 0 and 1.

    ``$pip install scikit-learn``

Use the following commands in Jupyter notebooks (if the libraries are already installed, this step could be skipped)

**Note: The aforementioned commands can be used in Jupyter Notebooks and can be ignored if the libraries have already been installed. Replace the '$ pip3' by '!pip' if you are using Google Colaboratory.**

Author:
"""""""

Harshit Bhavnani - harshit.bhavnani@gmail.com

License
"""""""
MIT
