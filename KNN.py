import os
import cv2 as cv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pretraitement import *
from extraction import *
import csv
from sklearn.model_selection import train_test_split

# Calcul la distance euclidienne entre deux points
def distance_euclidienne(point1, point2):
    distance = 0
    for col in range(len(point1)-1):
        distance = distance + ((point1[col]-point2[col])**2)
    return math.sqrt(distance)



#  Classifie les données de test en utilisant l'algorithme KNN
def KNN(donnees_entrainement, donnees_test, k):
    resultats = np.zeros([len(donnees_test), 1])
    for point in range(len(donnees_test)):
        distances = np.zeros([len(donnees_entrainement), 2])
        votes = np.zeros([k, 2])
        for ligne in range(len(donnees_entrainement)):
            distances[ligne, 0] = distance_euclidienne(donnees_entrainement[ligne], donnees_test[point])
            distances[ligne, 1] = donnees_entrainement[ligne][len(donnees_entrainement[ligne])-1]
        distances = distances[distances[:, 0].argsort()]
        distances = distances[:, :k]
        liste_voisins = []
        for i in range(k):
            liste_voisins.append(distances[i])
        
        for x in range(k):
            compteur = 0
            classe = liste_voisins[x][1]
            for y in range(k):
                if classe == liste_voisins[y][1]:
                    compteur = compteur + 1
            votes[x, 0] = classe
            votes[x, 1] = compteur
        votes = votes[votes[:, 1].argsort()]
        classe_majoritaire = votes[0, len(votes)-1, 0]
        resultats[point, 0] = classe_majoritaire
        
    return resultats


# Chargement et affichage des données à partir d'un fichier CSV
donnees = pd.read_csv("numbers.csv")
print(donnees)

# Séparation des variables indépendantes (X) et dépendantes (Y)
X = donnees.iloc[:, :-1].values
Y = donnees.iloc[:, -1].values

# Division des données en ensemble d'entraînement et de test
X_entrainement, X_test, Y_entrainement, Y_test = train_test_split(X, Y, test_size = 0.2)

# Préparation des données de test pour le modèle KNN
X_test = X_test[:, :-1]

# Prédiction des résultats avec KNN
y_pred = KNN(X_entrainement, X_test, 3)

# Affichage des résultats réels
print(Y_test)


# Taux de Classification & Taux d'erreur
somme = 0
for i in range(len(y_pred)):
    if Y_test[i] == y_pred[i]:
        somme += 1
taux_classification = (somme / len(y_pred)) * 100
taux_erreur = 100 - taux_classification
print("Le taux de classification est de:", taux_classification, "%")
print("Le taux d'erreur est de:", taux_erreur, "%")

# Affichage des prédictions KNN
print(y_pred)
