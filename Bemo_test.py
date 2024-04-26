#!/usr/bin/python3
# 2017.10.10 12:44:37 CST
# 2017.10.10 14:08:57 CST
import cv2
import numpy as np
import pytesseract

# Configuration du chemin d'accès à Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Charger l'image
img = cv2.imread("produit_basler_5.png")
W, H = img.shape[:2]
print('image chargée')

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
# Définir les paramètres en fonction de la taille de l'image
minDist = max(H, W) // 50
param1 = 200
param2 = 100
minRadius = max(H, W) // 20
maxRadius = max(H, W) // 2

# Detect circles
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, circles=None, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
# make canvas
canvas = img.copy()

# (3) Get the mean of centers and do offset
if circles is not None:
    circles = np.intp(np.array(circles))
    x,y,r = 0,0,0
    for ptx,pty, radius in circles[0]:
         cv2.circle(canvas, (ptx,pty), radius, (0,255,0), 1, 16)
         x += ptx
         y += pty
         r += radius
    cnt = len(circles[0])
    x = x//cnt
    y = y//cnt
    r = r//cnt
    x+=5
    y-=7

    # logPolar and rotate
    polar = cv2.logPolar(gray, (int(x), int(y)), 350, cv2.WARP_FILL_OUTLIERS)
    rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print('Image transformée')
            
    sharp_img_rotated = cv2.filter2D(rotated, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Utilisation de Pytesseract pour extraire le texte de l'image prétraitée
    extracted_text = pytesseract.image_to_string(sharp_img_rotated, lang='eng')
    
    # Affichage du texte extrait
    print("Texte extrait de l'image :")
    print(extracted_text)

    ## (6) Display the result  
    cv2.imshow("image éclairée", sharp_img_rotated)
    cv2.imshow("Image non éclairée", rotated)
    cv2.waitKey();cv2.destroyAllWindows()
    

else:
    print("No circles detected.")