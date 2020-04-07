import Classe.Data as data
import matplotlib.pyplot as plt

data = data.TuberculosisDataset()
data.loadData()

data, label = data.getTrainData()

images = data.reshape((-1, 96, 96))

plt.imshow(images[2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
