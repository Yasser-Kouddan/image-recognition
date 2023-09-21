import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Importation des données
donnees = pd.read_csv("numbers.csv")
print(donnees)

# Séparation des entrées et sorties
X = donnees.iloc[:,:-1].values
Y = donnees.iloc[:,-1].values

# Séparation en données d'entraînement et de test
X_entrainement, X_test, Y_entrainement, Y_test = train_test_split(X, Y, test_size = 0.2)

# Définition du modèle
modele_rn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(
    3,), max_iter=1000)

# Entraînement du modèle
modele_rn.fit(X_entrainement, Y_entrainement)

# Prédiction avec le modèle
y_pred = modele_rn.predict(X_test)
print(y_pred)

# Afficher les vraies sorties
print(Y_test)

# Taux de Classification et Taux d'erreur
score = modele_rn.score(X_test,Y_test)
print(f'Taux de classification: {score*100}%')
print(f"Taux d'erreur: {100-score*100}%")
