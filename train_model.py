import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("Aug_Lock_Flow_Training_Scaled.csv")

X = training_data_df.drop('Flow Direction', axis=1).values
Y = training_data_df[['Flow Direction']].values

# Define the model
model = Sequential()
model.add(Dense(20, input_dim=4, activation='softplus'))
model.add(Dense(40, activation='softplus'))
model.add(Dense(20, activation='softplus'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data_df = pd.read_csv("Aug_Lock_Flow_Testing_Scaled.csv")

X_test = test_data_df.drop('Flow Direction', axis=1).values
Y_test = test_data_df[['Flow Direction']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))