import Classe.Data as data
import Package.model as model
import matplotlib.pyplot as plt

donne = data.TuberculosisDataset()
donne.loadData()

dataTrain, labelTrain = donne.getTrainData()
dataTest, labelTest = donne.getTestData()

print("Training Images (Shape): ", dataTrain.shape);
print("Training Labels (Shape): ", labelTrain.shape);

KKN_Gen = model.kkn_initial(dataTrain, labelTrain, dataTest, labelTest)

print("conversion...")
images = dataTrain.reshape((-1, 192, 192))
print("Training Images (Shape): ", images.shape);

plt.imshow(images[2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
