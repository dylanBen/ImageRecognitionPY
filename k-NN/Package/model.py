import collectSample as collect
import matplotlib.pyplot as plt
from sklearn import neighbors
import joblib


def kkn_initial(dataTrain, labelTrain, dataTest, labelTest):
    errors = []
    opti = [100, 2]

    for k in range(2,19):
        print("knn en cours", k)
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(dataTrain, labelTrain)
        error = 100*(1 - knn.score(dataTest, labelTest))
        print("pourcentage d'erreurs :", round(error,2), "%")
        if opti[0] > error :
            opti[0] = error
            opti[1] = k
        errors.append(error)
    plt.plot(range(2,19), errors, 'o-')
    plt.show()
    print("KNN le plus fiable", opti[1], "  avec une erreur de", round(opti[0],2), "%")

    return neighbors.KNeighborsClassifier(opti[1])

def kkn_with_diff_number_image(knn, donne):
    bestKnn = {}
    errors = []
    opti = [100, 10]
    i = 10

    while i < 101:
        collect.launchSampling(i)
        print("nombre d'image de train", i)
        donne.loadData()

        dataTrain, labelTrain = donne.getTrainData()
        dataTest, labelTest = donne.getTestData()

        knn.fit(dataTrain, labelTrain)
        error = 100*(1 - knn.score(dataTest, labelTest))

        print("pourcentage d'erreurs :", round(error,2), "%")
        if opti[0] > error :
            opti[0] = error
            opti[1] = i
            bestKnn = knn
        errors.append(error)

        i = i + 10
    plt.plot(range(10,110,10), errors, 'o-')
    plt.show()
    print("le meilleur nombre d'image est de", opti[1], "  avec une erreur de", round(opti[0],2), "%")

    filename = 'k-NN/KNN.sav'
    joblib.dump(bestKnn, filename)
    print("model saved")

    return neighbors.KNeighborsClassifier(opti[1])
