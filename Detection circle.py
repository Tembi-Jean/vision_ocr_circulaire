import cv2
import numpy as np

# Charger l'image
img = cv2.imread("imgtest2.jpg")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))
canvas = img.copy()

# Centre de l'image
center_x = canvas.shape[1] // 2
center_y = canvas.shape[0] // 2

# Rayon maximum pour les cercles
max_radius = min(center_x, center_y) - 10

# Nombre de cercles à dessiner
num_circles = 10

# Espacement entre les cercles
spacing = max_radius // num_circles

# Dessiner les cercles
for i in range(1, num_circles + 1):
    radius = i * spacing
    cv2.circle(canvas, (center_x, center_y), radius, (0, 255, 0), 2)

# Afficher l'image avec les cercles dessinés
cv2.imshow("Cercles régulièrement espacés", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()