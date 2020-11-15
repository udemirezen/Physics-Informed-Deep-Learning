import tensorflow as tf
import keras.layers
from SimpleLSTMNetwork import prepare_data
import numpy as np
import matplotlib.pyplot as plt

# Read Data
points, temps, times, temps_mean, temps_std = prepare_data.read_data()
train_points, train_temps, eval_points, eval_temps, test_points, test_temps = prepare_data.train_val_test_app1(points,
                                                                                                               temps)
training_number = len(train_points)
eval_number = len(eval_points)
test_number = len(test_points)

# ***********************************************************

# Building The Model
rnn_model = tf.keras.Sequential()
rnn_model.add(tf.keras.layers.RepeatVector(100, input_shape=(5,)))
rnn_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
rnn_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
rnn_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
rnn_model.add(tf.keras.layers.Dense(100))
rnn_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["mse"])
# Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = rnn_model.fit(x=train_points, y=train_temps, batch_size=15, epochs=12, callbacks=[callback])
rnn_model.evaluate(x=eval_points, y=eval_temps)
rnn_model.summary()
# Determining the Test Error
prediction = rnn_model.predict(x=test_points)
# Test Error
test_errors = tf.keras.losses.MSE(test_temps, prediction)
print("Test Error = ", np.mean(test_errors))

# ***********************************************************

"""
Plotting the results
"""
# Test Errors
fig1, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_title('Test Errors Distribution')
ax1.boxplot(test_errors, vert=0)
plt.savefig('TEST ERRORS.png')

# ***********************************************************

# Predicting on all of the points
predictions = rnn_model.predict(x=points)
# Test on point number x (Vary from 1 to 1681)
point_number = 100
fig, ax = plt.subplots()
ax.plot(times, (temps[point_number].reshape(100, 1) * temps_std) + temps_mean)
ax.plot(times, (predictions[point_number].reshape(100, 1) * temps_std) + temps_mean)
print(tf.keras.losses.MSE(temps[point_number], predictions[point_number]))

# ***********************************************************

# Predicting on Test Points
test_preds = rnn_model.predict(x=test_points)
point_number = 900
fig, ax = plt.subplots()
ax.plot(times, test_temps[point_number].reshape(100, 1))
ax.plot(times, test_preds[point_number].reshape(100, 1))
print(tf.keras.losses.MSE(test_temps[point_number], test_preds[point_number]))
