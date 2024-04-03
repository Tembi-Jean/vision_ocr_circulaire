import cv2
import numpy as np

# Charger l'image
img = cv2.imread("imgtest3.jpg")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un flou gaussien pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Paramètres de détection des cercles
dp = 1  # Ratio de résolution de l'accumulateur
minDist = 50  # Distance minimale entre les centres des cercles détectés
param1 = 200  # Premier seuil pour le détecteur de bord de Canny
param2 = 30   # Seuil pour le vote de l'accumulateur
minRadius = 0  # Rayon minimum du cercle
maxRadius = 0  # Rayon maximum du cercle (0 pour ne pas limiter)

# Détection des cercles
circles = cv2.HoughCircles(blurred, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

# Dessiner les cercles détectés sur l'image originale
if circles is not None:
    circles = np.int0(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(img, center, radius, (0, 255, 0), 2)

# Afficher l'image avec les cercles détectés
cv2.imshow("Cercles détectés", img)
cv2.waitKey(0)
cv2.destroyAllWindows()