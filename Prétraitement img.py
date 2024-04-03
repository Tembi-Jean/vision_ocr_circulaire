import cv2

# Charger l'image
img = cv2.imread("imgtest3.jpg")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer une égalisation d'histogramme pour améliorer le contraste
eq_gray = cv2.equalizeHist(gray)

# Appliquer un filtrage gaussien pour réduire le bruit
blurred = cv2.GaussianBlur(eq_gray, (5, 5), 0)

# Afficher les images
cv2.imshow("Image originale", img)
cv2.imshow("Image en niveaux de gris", gray)
cv2.imshow("Image après égalisation d'histogramme", eq_gray)
cv2.imshow("Image après filtrage gaussien", blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()