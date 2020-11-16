import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def read_data():
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

    times = genfromtxt('time.csv', delimiter=',')

    all = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11), axis=0)

    points = all[:, 0:5]
    temps = all[:, 5:]

    points, temps, temps_mean, temps_std = normalize_data(points, temps)
    return points, temps, times, temps_mean, temps_std


def normalize_data(points, temps):
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

    return points, temps, temps_mean, temps_std


def train_val_test_app1(points, temps):
    """
    Approach 1. Use first 9 for Training, 1 for Eval and 1 for Test
    """
    train_points = points[: 9 * 1681, :]
    train_temps = temps[: 9 * 1681, :]
    eval_points = points[9 * 1681: 10 * 1681, :]
    eval_temps = temps[9 * 1681: 10 * 1681, :]
    test_points = points[10 * 1681:, :]
    test_temps = temps[10 * 1681:, :]
    return train_points, train_temps, eval_points, eval_temps, test_points, test_temps


def train_val_test_app2(points, temps):
    """
    Approach 2. K-Means Clustering
    """
    # Cluster just based on the coordinates (0:3)
    point = points[:, 0:3]
    kmeans = KMeans(n_clusters=10)
    kmeans = kmeans.fit(point)
    labels = kmeans.predict(point)

    """
    ## In order to plot the points with their labels, uncomment this part :)
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
    """
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

    return train_points, train_temps, eval_points, eval_temps, test_points, test_temps
