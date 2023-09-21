import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Binarisation de l'image
def binariser_image(img):
    hauteur, largeur = img.shape
    somme = 0
    for i in range(hauteur):
        for j in range(largeur):
            somme += img[i,j]
    seuil = somme/(largeur*hauteur)
    img_bin = img.copy()
    img_bin = img_bin/255
    for i in range(hauteur):
        for j in range(largeur):
            if img[i,j] < seuil:
                img_bin[i,j] = 0
            else: 
                img_bin[i,j] = 255
    return img_bin

# Inverser l'image
def inverser_image(img):
    img_inv = binariser_image(img)
    hauteur, largeur = img_inv.shape
    for i in range(hauteur):
        for j in range(largeur):
            if img_inv[i,j] == 1:
                img_inv[i,j] = 0
            else: 
                img_inv[i,j] = 1
    return img_inv

# redimensionner l'image
def redimensionner_image(img):
    img_res = cv.resize(img, (500, 500))
    return img_res
