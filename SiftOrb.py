import cv2
import numpy as np


image = cv2.imread('Suka.jpg', cv2.IMREAD_GRAYSCALE)

# Criar o objeto SIFT
sift = cv2.SIFT_create()

# Detectar e computar características
keypoints, descriptors = sift.detectAndCompute(image, None)

# Desenhar keypoints na imagem
imagem_sift = cv2.drawKeypoints(image, keypoints, None)

# Exibir resultados
cv2.imshow('SIFT', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Criar o objeto ORB
orb = cv2.ORB_create()

# Detectar e computar características
keypoints, descriptors = orb.detectAndCompute(image, None)

# Desenhar keypoints na imagem
imagem_orb = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Exibir resultados
cv2.imshow('ORB', imagem_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()
