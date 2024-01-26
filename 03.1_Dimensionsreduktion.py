import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn import datasets
from sklearn.decomposition import PCA

data, shape = datasets.make_swiss_roll(n_samples=1000, noise=0.0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d') #erzeugt leeres 3D-Koordinatensystem
ax.scatter(data[:,0], data[:,1], data[:,2], c=shape)
plt.show()

#In PY funktioniert die Darstellung besser



