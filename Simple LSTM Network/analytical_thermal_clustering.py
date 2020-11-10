# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import keras.layers
from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

"""### **Load Dataset**"""

p1 = genfromtxt('T_q100_v200_T25_D100.csv', delimiter=',')
p2 = genfromtxt('T_q100_v300_T25_D100.csv', delimiter=',')
p3 = genfromtxt('T_q100_v400_T25_D100.csv', delimiter=',')
p4 = genfromtxt('T_q100_v400_T25_D100.csv', delimiter=',')
p5 = genfromtxt('T_q150_v400_T25_D100.csv', delimiter=',')
p6 = genfromtxt('T_q200_v500_T25_D100.csv', delimiter=',')
p7 = genfromtxt('T_q200_v800_T25_D100.csv', delimiter=',')
p8 = genfromtxt('T_q200_v1000_T25_D100.csv', delimiter=',')
p9 = genfromtxt('T_q200_v1200_T25_D100.csv', delimiter=',')
p10 = genfromtxt('T_q200_v1500_T25_D100.csv', delimiter=',')
p11 = genfromtxt('T_q300_v1000_T25_D100.csv', delimiter=',')

times = genfromtxt('time..csv', delimiter=',')

all = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11), axis=0)

points = all[:, 0:5]
temps = all[:, 5:]

"""### **Normalizing the Data**
We normalize every column of our data, except the x axis because it is common (we can just omit it)
"""

# Normalizing the temperatures:
temps_mean = np.zeros(100)
temps_std = np.zeros(100)
for i in range(100):
    temps_mean[i] = temps[:, i].mean()
    temps_std[i] = temps[:, i].std()
    temps[:, i] = (temps[:, i] - temps_mean[i]) / temps_std[i]

temps_std = temps_std.reshape((100, 1))
temps_mean = temps_mean.reshape((100, 1))

points_mean = np.zeros(5)
points_std = np.zeros(5)
# Normalizing the points
for i in range(4):
    i = i + 1
    points_mean[i] = points[:, i].mean()
    points_std[i] = points[:, i].std()
    points[:, i] = (points[:, i] - points_mean[i]) / points_std[i]

"""# **2. K-Means**"""

# Cluster just based on the coordinates (0:3)
point = points[:, 0:3]
kmeans = KMeans(n_clusters=10)
kmeans = kmeans.fit(point)
labels = kmeans.predict(point)

fig = plt.figure(1, figsize=(10, 10))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(point[:, 0], point[:, 1], point[:, 2],
           c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.dist = 12
fig.show()

# Nice Pythonic way to get the indices of the points for each corresponding cluster
mydict = {i: np.where(labels == i)[0] for i in range(kmeans.n_clusters)}
# Transform this dictionary into list (if you need a list as result)
dictlist = []
for key, value in mydict.items():
    dictlist.append(value)

training_indices = []
evaluation_indices = []
test_indices = []
for l in dictlist:
    training_indices.extend(l[0: int(len(l) * 0.8)])
    evaluation_indices.extend(l[int(len(l) * 0.8): int(len(l) * 0.8) + int(len(l) * 0.1)])
    test_indices.extend(l[int(len(l) * 0.8) + int(len(l) * 0.1):])

training_number = len(training_indices)
eval_number = len(evaluation_indices)
test_number = len(test_indices)

concat = np.concatenate((points, temps), axis=1)
training = np.asarray([concat[i] for i in training_indices])
evaluation = np.asarray([concat[i] for i in evaluation_indices])
testing = np.asarray([concat[i] for i in test_indices])
train_points = training[:, 0:5]
train_temps = training[:, 5:]
eval_points = evaluation[:, 0:5]
eval_temps = evaluation[:, 5:]
test_points = testing[:, 0:5]
test_temps = testing[:, 5:]

"""### **Building The Model**"""

# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

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

"""Run the following two cells just in case you want to visualize in tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

rnn_model.evaluate(x=eval_points, y=eval_temps)

rnn_model.summary()

"""### **Determining the Test Error**"""

prediction = rnn_model.predict(x=test_points)

# Test Error
test_errors = tf.keras.losses.MSE(test_temps, prediction)
np.mean(test_errors)

fig1, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_title('Test Errors Distribution')
ax1.boxplot(test_errors, vert=0)
plt.savefig('TEST ERRORS.png')

"""## **Predicting on all of the points**

---
"""

predictions = rnn_model.predict(x=points)

# Test on point number x (Vary from 1 to 1681)
point_number = 100
fig, ax = plt.subplots()
ax.plot(times, (temps[point_number].reshape(100, 1) * temps_std) + temps_mean)
ax.plot(times, (predictions[point_number].reshape(100, 1) * temps_std) + temps_mean)
print(tf.keras.losses.MSE(temps[point_number], predictions[point_number]))

"""## **Predicting on Test Points**"""

test_preds = rnn_model.predict(x=test_points)

point_number = 900
fig, ax = plt.subplots()
ax.plot(times, test_temps[point_number].reshape(100, 1))
ax.plot(times, test_preds[point_number].reshape(100, 1))
print(tf.keras.losses.MSE(test_temps[point_number], test_preds[point_number]))
plt.savefig('3t.png')
