import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
from datetime import timedelta, date
import keras
from keras import layers
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Nadam


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


tkr = yf.Ticker('SPY')
tf = tkr.history(period="5y")
ttf = tf.iloc[::-1]
ta = ttf.rename(columns={'Close': 'Price'})
ta['OneDayChange'] = ta['Price'].shift(-1) - ta['Price'].shift(-2)
ta['Derivative'] = ta['Price'].shift(-1) - 2*ta['Price'].shift(-2) + ta['Price'].shift(-3)

tx = ta[['Price','OneDayChange','Derivative']]


df = tx.dropna() # droping the rows that contain NaN.




print(df)    
print(df.shape)

# convert the DF to a flat (one-dimensional) NumPy array.
arr = df.values.flatten()

#the size (height) of the sliding window
sw_height = 10 #the number of points (days) you want to include in a sample
#the step size of the sliding window
sw_step = len(df.columns) #the number of columns in the DataFrame, including Y and X columns
#space for the sliding window movement
rows = len(df) #the number of rows in the DataFrame

# create an indexer, an array 
# of indices that map the original array to the desired array.
# create an indexer that reshapes the flat array so the first column in the new array contains the target variable 
# (the price for a certain day) while the other columns (designed for holding
#  the input variables) are formed from the subsequent elements of the array found within the sliding window.

idx = np.arange(sw_step*sw_height)[None, :] + sw_step*np.arange(rows-sw_height+1)[:, None]


# Once you have the indexer, you can reshape the original array.

arr = arr[idx]


# print(arr.shape) # To verify that these steps have worked as expected, look at the shape of the new array.


target = arr[:, 0]

# Then, you save all the other columns (29, in this example) in the features array.

features = arr[:, 1:]

print("target array shape: ", target.shape, "\nfeatures array shape: ", features.shape)

# Assuming you have n number samples in this example,
# you might put the last 1,000 samples for a testing set 
# into the training set. To do that, the dataset must start from n (where n = number of samples remove first n1) 
# as follows: (#ex 1246 = 246, or 3122 = 122)

x_train = features[246:]
y_train = target[246:]

# Put the first 246 – 10 samples (allowing for the height of the sliding window) 
# into the testing set that you will use to evaluate the model.

x_eval = features[:246-sw_height]
y_eval = target[:246-sw_height]


# Subtract the 10 last samples from the testing set, because they partially include
#  data from the first samples in the training set. 
# Now, make sure that the split has been done as expected.

print(x_train.shape, x_eval.shape, y_train.shape, y_eval.shape)

model = keras.Sequential([
  layers.Dense(550, input_shape=x_train.shape[1:], activation="relu"),
  layers.Dense(325, activation="relu"),
  layers.Dense(300, activation="relu"),
  layers.Dense(150, activation="relu"),
  layers.Dense(350, activation="relu"),
  layers.Dense(150, activation="relu"),
  layers.Dense(150, activation="relu"),
  layers.Dense(100, activation="relu"),
  layers.Dense(100, activation="relu"),
  layers.Dense(175, activation="relu"),
  layers.Dense(125, activation="relu"),
  layers.Dense(275, activation="relu"),
  layers.Dense(325, activation="relu"),
  layers.Dense(125, activation="relu"),
  layers.Dense(20, activation="relu"),
  layers.Dense(115, activation="relu"),
  layers.Dense(170, activation="relu"),
  layers.Dense(100, activation="relu"),
  layers.Dense(100, activation="relu"),
  layers.Dense(75, activation="relu"),
  layers.Dense(525, activation="relu"),
  layers.Dense(75, activation="relu"),
  layers.Dense(225, activation="relu"),
  layers.Dense(25, activation="tanh"),
  layers.Dense(20, activation="tanh"),
  layers.Dense(15, activation="tanh"),
  layers.Dense(10, activation="tanh"),
  layers.Dense(100, activation="relu"),
  layers.Dense(85, activation="relu"),
  layers.Dense(125, activation="relu"),
  layers.Dense(55, activation="relu"),
  layers.Dense(125, activation="relu"),
  layers.Dense(50, activation="relu"),
  layers.Dense(55, activation="relu"),
  layers.Dense(25, activation="relu"),
  layers.Dense(75, activation="relu"),
  layers.Dense(225, activation="relu"),
  # layers.Dense(125, activation="softmax"),
  # layers.Dense(125, activation="softmax"),
  # layers.Dense(125, activation="softmax"),
  # layers.Dense(125, activation="softmax"),


  layers.Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-09), loss="mse", metrics=["mae"])


model.summary()


model.fit(x_train, y_train, verbose=1, epochs=1500)

mse, mae = model.evaluate(x_eval, y_eval, verbose=0)
print("MSE: ", mse, "\nMAE: ", mae)

##remember that each epoch will have its own mse and mae
##this prediction will be for overall model prediction capabilities on data used
predictions = model.predict(x_eval[0:5])
print("Actual: ", y_eval[0:5], "\nPredictions: ", predictions.flatten())
