# from sklearn.utils import *x
import numpy as np
from fs.osfs import OSFS
from PIL import Image, ImageFilter
import os

TRAIN_IMAGES_DIR = './data/train';
TEST_IMAGES_DIR = './data/test';

def loadImages(dataDir):
    images = [];
    labels = [];

    files = OSFS(".")
    for file in files.listdir(dataDir):
        filePath = dataDir + "/" + file

        if (not file.lower().endswith(".jpeg")):
            if (not files.getinfo(filePath).is_dir):
                continue
            imagesSous, labelsSous = loadImages(filePath)
            images = images + imagesSous
            labels = labels + labelsSous
            continue

        images.append(imagePrepare(filePath))

        #0 pour poumon normal et 1 pour poumon malade
        if dataDir.upper().endswith("PNEUMONIA"):
            labels.append(1)
        else:
            labels.append(0)

    files.close()
    return [images, labels]

def imagePrepare(path):
    """
    :param path: chemin de l'image à convertir
    :return: l'image sous le format 192*192 pixel (36 864)
    """
    im = Image.open(path).convert("L")
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new("L", (192,192), 255)

    if width > height:
        nheight = int(round((192.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1

        img = im.resize((192, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((192 - nheight) / 2), 0))
        newImage.paste(img, (0, wtop))
    else:
        # nwidth = int(round((20.0 / height * width), 0))
        nwidth = int(round((192.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1

        img = im.resize((nwidth, 192), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((192 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())

    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return np.array(tva)

#Helper class to handle loading training and test data. */
class TuberculosisDataset:

    def __init__(self): # Notre méthode constructeur
        self.trainData = [];
        self.testData = [];

    # /** Loads training and test data. */
    def loadData(self):
        print('Loading images...');

        #permet de revenir sur le dossier parent
        os.chdir("..")

        self.trainData = loadImages(TRAIN_IMAGES_DIR);

        self.testData = loadImages(TEST_IMAGES_DIR);

        print('Images loaded successfully.')

    def getTrainData(self):
        return [np.array(self.trainData[0]), np.array(self.trainData[1])]
        # return Bunch(images=self.trainData[0], labels=np.array(self.trainData[1]))

    def getTestData(self):
        return [np.array(self.testData[0]), np.array(self.testData[1])]
        # return Bunch(images=self.testData[0], labels=np.array(self.testData[1]))
