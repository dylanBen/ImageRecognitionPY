import matplotlib.pyplot as plt
from sklearn import neighbors
import joblib

opti = [100, 2]
bestKnn = {}

def kkn_initial(dataTrain, labelTrain, dataTest, labelTest):
    errors = []

    for k in range(2,19):
        print("knn en cours", k)
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(dataTrain, labelTrain)
        error = 100*(1 - knn.score(dataTest, labelTest))
        print("pourcentage d'erreurs :", round(error,2), "%")
        if opti[0] > error :
            opti[0] = error
            opti[1] = k
            bestKnn = knn
        errors.append(error)
    plt.plot(range(2,19), errors, 'o-')
    plt.show()
    print("KNN le plus fiable", opti[1], "  avec une erreur de", round(opti[0],2), "%")

    filename = 'k-NN/KNN.sav'
    joblib.dump(bestKnn, filename)
    print("model saved")

    return neighbors.KNeighborsClassifier(opti[1])
