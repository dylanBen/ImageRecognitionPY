import Classe.Data as data

data = data.TuberculosisDataset()
data.loadData()

data, label = data.getTrainData()
