import os
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pretraitement import *
from extraction import *
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Chemin vers le répertoire de données
data_dir = 'dataset'
# Liste des classes
Classes = ['classe 0', 'classe 1', 'classe 2', 'classe 3', 'classe 4' , 'classe 5', 'classe 6', 'classe 7', 'classe 8', 'classe 9']

# En-tête pour le fichier CSV
header = ["phi0", "phi1", "phi2", "phi3", "phi4", "phi5", "phi6","classe"]
data = []

# Boucle pour traiter les images de chaque classe
for classe in Classes:

    # Récupération du numéro de la classe en cours
    class_num = Classes.index(classe)

    # Chemin vers le répertoire contenant les images de la classe en cours
    path = os.path.join(data_dir, classe)

    # Boucle pour traiter chaque image de la classe en cours
    for img in os.listdir(path):
        img = cv.imread(os.path.join(path, img),0)
        img_bin = binariser_image(img)
        img_resize = redimensionner_image(img_bin)
        moments_hu(img_bin, class_num)
        data.append(moments_hu(img_bin, class_num))

# Écriture des données dans un fichier CSV
with open('numbers_v2.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # Écriture de l'en-tête
            writer.writerow(header)

            # Écriture de plusieurs lignes
            writer.writerows(data)


#Lecture et affichage des données depuis le fichier csv
data = pd.read_csv("numbers.csv")
data

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
Y

plt.scatter(data['phi1'], Y)
plt.show()

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)

score = knn.score(X_test,Y_test)
print(f'taux de classification: {score*100}%')

# Affichage des prédictions
Y_pred = knn.predict(X_test)
print(Y_pred)

# Affichage des classes réelles pour les données de test
print(Y_test)

