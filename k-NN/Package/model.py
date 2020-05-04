import matplotlib.pyplot as plt
from sklearn import neighbors

opti = [100, 2]

def kkn_initial(dataTrain, labelTrain, dataTest, labelTest):
    errors = []

    for k in range(2,15):
        print("knn en cours", k)
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(dataTrain, labelTrain)
        error = 100*(1 - knn.score(dataTest, labelTest))
        print("pourcentage d'erreurs :", round(error,2), "%")
        if opti[0] > error :
            opti[0] = error
            opti[1] = k
        errors.append(error)
    plt.plot(range(2,15), errors, 'o-')
    plt.show()
    print("KNN le plus fiable", opti[1], "  avec une erreur de", round(opti[0],2), "%")

    return neighbors.KNeighborsClassifier(opti[1])
