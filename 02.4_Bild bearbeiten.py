import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

img = plt.imread('data/papa_color.png')
plt.imshow(img)
plt.show()      #Originalbild


data = np.reshape(img, (512*512, 3))
model = KMeans(10)      #Anzahl der Farben bestimmen
model.fit(data)
data_reduced = model.cluster_centers_[model.predict(data)]
img16 = np.reshape(data_reduced, (512, 512, 3))
plt.imshow(img16)
plt.show()      #bearbeitetes Bild lädt eine Zeit

