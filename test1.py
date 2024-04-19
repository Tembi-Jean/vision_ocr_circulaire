#!/usr/bin/python3
import cv2
import numpy as np
import pytesseract

# Configuration du chemin d'accès à Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Charger l'image
img = cv2.imread("imgtest4.jpg")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre de netteté à l'image en niveaux de gris
sharp_img = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

# Detect circles
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=3, circles=None, param1=200, param2=100, minRadius = 200, maxRadius=0 )

# make canvas
canvas = img.copy()

# Get the mean of centers and do offset
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

    # Draw the labels in red
    for r in range(100, r, 20):
        cv2.circle(canvas, (x,y), r, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x,y), 3, (0,0,255), -1)

    # logPolar and rotate
    polar = cv2.logPolar(gray, (int(x), int(y)), 120, cv2.WARP_FILL_OUTLIERS )
    rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Appliquer un filtre de netteté à l'image en niveaux de gris
    sharp_img_rotated = cv2.filter2D(rotated, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Initialiser les coordonnées du rectangle
    roi_points = []
    drawing = False

    # Fonction de rappel de la souris
    def select_roi(event, x, y, flags, param):
        global roi_points, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_points = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_points.append((x, y))

    # Créer une fenêtre et attacher la fonction de rappel de la souris
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_roi)

    while True:
        # Dessiner le rectangle de sélection sur l'image
        if len(roi_points) == 2:
            cv2.rectangle(rotated, roi_points[0], roi_points[1], (0, 255, 0), 2)

        # Afficher l'image
        cv2.imshow('image', rotated)

        # Appuyez sur 'q' pour quitter et extraire la ROI
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Extraire la ROI de l'image
    if len(roi_points) == 2:
        roi = rotated[roi_points[0][1]:roi_points[1][1], roi_points[0][0]:roi_points[1][0]]
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Utilisation de Pytesseract pour extraire le texte de l'image prétraitée
    extracted_text = pytesseract.image_to_string(roi, lang='eng')
    
    # Affichage du texte extrait
    print("Texte extrait de l'image :")
    print(extracted_text)

else:
    print("No circles detected.")