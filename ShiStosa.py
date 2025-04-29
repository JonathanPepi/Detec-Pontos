import cv2
import numpy as np


image = cv2.imread('Suka.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar o detector Shi-Tomasi
cantos = cv2.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.05, minDistance=10)
cantos = np.int32(cantos)

# Marcar os cantos na imagem
for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Exibir resultados
cv2.imshow('Shi-Tomasi', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
