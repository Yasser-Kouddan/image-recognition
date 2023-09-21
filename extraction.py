import cv2 as cv
import numpy as np
import math


########################## Moment de Hu #####################


#Calculer le moment d'ordre p et q de l'image binaire donnée
def calculer_moment_p_q(image_binaire, p, q):
    somme = 0
    hauteur, largeur = image_binaire.shape
    for x in range(hauteur):
        for y in range(largeur):
            somme += (x**p)*(y**q)*image_binaire[x, y]
    return somme


def moments_centraux_pq(image_binaire, p, q):
    moment = 0
    hauteur, largeur = image_binaire.shape
    a = calculer_moment_p_q(image_binaire, 1, 0)/calculer_moment_p_q(image_binaire, 0, 0)
    b = calculer_moment_p_q(image_binaire, 0, 1)/calculer_moment_p_q(image_binaire, 0, 0)
    for x in range(largeur):
        for y in range(hauteur):
            moment += (((x-a)**p)*((y-b)**q))*image_binaire[x, y]
    return moment

# Les moments centraux normalisés
def moment_central_normalise_pq(image_binaire, p, q):
    lambda_ = ((p+q+1)/2) + 1
    moment_central = calculer_moment_p_q(image_binaire, p, q)/(calculer_moment_p_q(image_binaire, 0, 0)**lambda_)
    return moment_central


def moments_hu(ib, num_classe):
    phi = np.zeros([8], dtype=float)
    
    A02 = moment_central_normalise_pq(ib, 0, 2)
    A03 = moment_central_normalise_pq(ib, 0, 3)
    A11 = moment_central_normalise_pq(ib, 1, 1)
    A12 = moment_central_normalise_pq(ib, 1, 2)
    A20 = moment_central_normalise_pq(ib, 2, 0)
    A21 = moment_central_normalise_pq(ib, 2, 1)
    A30 = moment_central_normalise_pq(ib, 3, 0)
    
    phi[0] = A20 - A02
    phi[1] = (A20 - A02)**2 + (4*A11)**2
    phi[2] = (A30 - A12)**2 +(3*A12 - A03)**2
    phi[3] = (A30 + A12)**2 + (A21+A03)**2
    
    phi[4] = (A30 - 3*A12)*(A30+A12)*((A30+A12)**2 - 3*(A21+A03)**2) +(3*A21-A03)*(A21+A03)*(3*(A30+A12)**2-((A21+A03)**2))
   
    phi[5] = (A20-A02)*((A30+A12)**2-(A21+A03)**2)+4*A11*(A30+A12)*(A21+A03)
   
    phi[6] = (3*A21-A30)*(A30+A12)*((A30+A12)**2-3*((A21+A03)**2))+(3*A12-A03)*(A21+A03)*((3*(A30+A12)**2)-((A21+A03)**2))
    phi[7] = num_classe
    return phi;



####################### Moment de tchibichef ###################


# Calcule la norme au carré d'un polynôme de Legendre
def norme_carree_polynomial(n, N):
    # limites pour les boucles
    limite1 = n + N
    limite2 = N - n - 1

    somme1 = 1
    for i in range(1, limite1 + 1):
        somme1 = somme1 * i

    somme2 = 1
    for i in range(1, limite2 + 1):
        somme2 = somme2 * i

    somme3 = 2 * n + 1
    # calcul final
    norme_carree = somme1 / (somme3 * somme2)
    return norme_carree

# Calcule le pochhammer d'un nombre avec un exposant k
def pochhammer(a, k):
    a = 1
    for i in range(1, k + 1):
        a = a * (a + (i - 1))
    return a

# Calcule le polynôme de Legendre pour un n donné
def polynome_legendre(n, M, x):
    # calcul des termes intermédiaires
    t1 = pochhammer(1-M, n)
    t2 = math.sqrt(norme_carree_polynomial(n, M))
    somme = 0
    for i in range(0, n + 1):
        t3 = pochhammer(-n, i)
        t4 = pochhammer(-x, i)
        t5 = pochhammer(1 + n, i)
        t6 = math.factorial(i)**2
        t7 = pochhammer(1-M, i)
        somme += ((t3 * t4 * t5) / (t6 * t7))
    # calcul final
    polynome = (t1 / t2) * somme        
    return polynome

# Calcule le moment normalisé d'une image
def moment_normalise(img, n, m):
    somme = 0
    M, N = img.shape
    for i in range(0, M):
        for j in range(0, N):
            t1 = polynome_legendre(i, n, M)
            t2 = polynome_legendre(j, m, N)
            somme += (t1 * t2 * img[i, j])
    return somme

# Calcule les coefficients de Tchebichef d'une image
def coefficients_tchebichef(img, ordre):
    # liste pour stocker les coefficients
    T = []
    for i in range(0, ordre + 1):
        for j in range(0, ordre + 1):
            if (i + j) <= ordre:
                T.append(moment_normalise(img, i, j))
    return T

