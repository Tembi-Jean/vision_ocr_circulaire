"""
#!/usr/bin/python3
# 2017.10.10 12:44:37 CST
# 2017.10.10 14:08:57 CST
import cv2
import numpy as np

##(1) Read and resize the original image(too big)
img = cv2.imread("imgbasler3.png")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer une égalisation d'histogramme pour améliorer le contraste
eq_gray = cv2.equalizeHist(gray)

## (2) Detect circles
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=3, circles=None, param1=200, param2=100, minRadius = 200, maxRadius=0 )

## make canvas
canvas = img.copy()

## (3) Get the mean of centers and do offset
if circles is not None:
    circles = np.int0(np.array(circles))
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

    ## (4) Draw the labels in red
    for r in range(100, r, 20):
        cv2.circle(canvas, (x,y), r, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x,y), 3, (0,0,255), -1)

    ## (5) logPolar and rotate
    polar = cv2.logPolar(img, (int(x), int(y)), 80, cv2.WARP_FILL_OUTLIERS)
    rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ## (6) Display the result
    cv2.imshow("Canvas", canvas)

    cv2.imshow("polar", polar)
    cv2.imshow("rotated", rotated)

    cv2.waitKey();cv2.destroyAllWindows()

else:
    print("No circles detected.")
"""
"""
#!/usr/bin/python3
# 2017.10.10 12:44:37 CST
# 2017.10.10 14:08:57 CST
import cv2
import numpy as np

def main():
    ##(1) Lire et redimensionner l'image originale (trop grande)
    img = cv2.imread("imgbasler3.png")
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer une égalisation d'histogramme pour améliorer le contraste
    eq_gray = cv2.equalizeHist(gray)

    ## (2) Détecter les cercles
    circles = cv2.HoughCircles(eq_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=3, circles=None, param1=200, param2=100, minRadius=200, maxRadius=0)

    ## Créer un canevas
    canvas = img.copy()

    ## (3) Obtenir la moyenne des centres et effectuer le décalage
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = 0, 0, 0
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(canvas, center, radius, (0, 255, 0), 1, 16)
            x += circle[0]
            y += circle[1]
            r += radius
        cnt = circles.shape[1]
        x = int(x // cnt)
        y =int(y // cnt)
        r = int(r // cnt)
        x += 5
        y -= 7

        ## (4) Dessiner les étiquettes en rouge
        for r in range(100, r, 20):
            cv2.circle(canvas, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)

        ## (5) LogPolar et rotation
        polar = cv2.logPolar(img, (x, y), 80, cv2.WARP_FILL_OUTLIERS)
        rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ## (6) Afficher le résultat
        cv2.imshow("Canvas", canvas)
        cv2.imshow("polar", polar)
        cv2.imshow("rotated", rotated)

        cv2.waitKey()
        cv2.destroyAllWindows()

    else:
        print("Aucun cercle détecté.")

if __name__ == "__main__":
    main()
"""

#"""
import cv2  # Importe le module OpenCV pour le traitement d'images
import numpy as np  # Importe le module NumPy pour les opérations numériques sur les tableaux
from image_processing import *
from image_processing import convert_to_grayscale, reduce_noise, binarize_image, enhance_contrast, morphological_operations, extract_text_regions, resize_image, preprocess_image,preprocess_and_binarize

import pytesseract

def read_and_resize_image(file_path, scale_factor=0.25):
    """
    Read and resize the image from the given file path.
    Args:
        file_path: File path of the image.
        scale_factor: Factor by which the image should be resized.
    Returns:
        Resized image.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        img = cv2.imread(file_path)  # Lecture de l'image à partir du chemin du fichier
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)  # Redimensionne l'image selon le facteur d'échelle
        return img  # Renvoie l'image redimensionnée
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in read_and_resize_image: {e}")  # Affiche l'erreur s'il y en a une
        return None  # Renvoie None en cas d'erreur

def detect_circles(image):
    """
    Detect circles in the given grayscale image.
    Args:
        image: Grayscale image.
    Returns:
        Detected circles.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertit l'image en niveaux de gris
        eq_gray = cv2.equalizeHist(gray)  # Applique une égalisation d'histogramme pour améliorer le contraste
        circles = cv2.HoughCircles(eq_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=3, circles=None, param1=200, param2=100, minRadius=200, maxRadius=0)  # Détecte les cercles dans l'image
        return circles  # Renvoie les cercles détectés
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in detect_circles: {e}")  # Affiche l'erreur s'il y en a une
        return None  # Renvoie None en cas d'erreur

def draw_circles(canvas, circles):
    """
    Draw circles on the given canvas.
    Args:
        canvas: Image to draw on.
        circles: Circles to draw.
    Returns:
        Image with circles drawn on it.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        if circles is not None:  # Vérifie si des cercles ont été détectés
            circles = np.uint16(np.around(circles))  # Convertit les cercles en entiers non signés
            for circle in circles[0, :]:  # Parcourt les cercles détectés
                center = (circle[0], circle[1])  # Coordonnées du centre du cercle
                radius = circle[2]  # Rayon du cercle
                cv2.circle(canvas, center, radius, (0, 255, 0), 1, 16)  # Dessine le cercle sur le canevas
        return canvas  # Renvoie le canevas avec les cercles dessinés
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in draw_circles: {e}")  # Affiche l'erreur s'il y en a une
        return None  # Renvoie None en cas d'erreur

def get_average_circle(circles):
    """
    Calculate the average circle from detected circles.
    Args:
        circles: Detected circles.
    Returns:
        Average circle parameters: center x, center y, radius.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        if circles is not None:  # Vérifie si des cercles ont été détectés
            circles = np.uint16(np.around(circles))  # Convertit les cercles en entiers non signés
            x, y, r = 0, 0, 0  # Initialise les variables pour les coordonnées moyennes et le rayon
            for circle in circles[0, :]:  # Parcourt les cercles détectés
                x += circle[0]  # Somme des coordonnées x des centres des cercles
                y += circle[1]  # Somme des coordonnées y des centres des cercles
                r += circle[2]  # Somme des rayons des cercles
            cnt = circles.shape[1]  # Nombre total de cercles détectés
            x = int(x / cnt) + 5  # Coordonnée x moyenne du centre des cercles avec un décalage
            y = int(y / cnt) - 7  # Coordonnée y moyenne du centre des cercles avec un décalage
            r = int(r / cnt)  # Rayon moyen des cercles
            return x, y, r  # Renvoie les coordonnées moyennes et le rayon du cercle
        else:
            return None  # Renvoie None si aucun cercle n'a été détecté
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in get_average_circle: {e}")  # Affiche l'erreur s'il y en a une
        return None  # Renvoie None en cas d'erreur

def draw_labels(canvas, center_and_radius):
    """
    Draw labels on the canvas around the average circle.
    Args:
        canvas: Image to draw on.
        center_and_radius: Tuple containing center coordinates and radius.
    Returns:
        Image with labels drawn on it.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        if center_and_radius is not None:  # Vérifie si les coordonnées du centre et le rayon sont valides
            x, y, radius = center_and_radius  # Récupère les coordonnées du centre et le rayon
            for r in range(100, radius, 20):  # Parcourt les rayons pour dessiner les cercles autour du centre
                cv2.circle(canvas, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # Dessine les cercles
                cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)  # Dessine un petit cercle au centre
        return canvas  # Renvoie le canevas avec les labels dessinés
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in draw_labels: {e}")  # Affiche l'erreur s'il y en a une
        return None  # Renvoie None en cas d'erreur

def logpolar_and_rotate(image, center):
    """
    Apply logPolar transformation and rotate the image.
    Args:
        image: Image to transform.
        center: Center point for logPolar transformation.
    Returns:
        Transformed image.
    """
    try:  # Début du bloc try-except pour capturer les erreurs
        if center is not None:  # Vérifie si le centre est valide
            polar = cv2.logPolar(image, center, 80, flags=cv2.WARP_FILL_OUTLIERS)  # Applique la transformation logPolar
            rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Effectue une rotation de 90 degrés dans le sens anti-horaire
            return polar, rotated  # Renvoie l'image transformée
        else:
            return None, None  # Renvoie None si le centre est None
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in logpolar_and_rotate: {e}")  # Affiche l'erreur s'il y en a une
        return None, None  # Renvoie None en cas d'erreur

# Ajoutez cette fonction après les autres fonctions de votre script
def detect_and_read_text(image):
    """
    Detect text in the given image using OCR.
    Args:
        image: Image containing text.
    Returns:
        Detected text.
    """
    try:
        # Convertit l'image en niveaux de gris pour une meilleure détection du texte
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Applique un filtre flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Utilise OCR pour détecter et lire le texte dans l'image
        text = pytesseract.image_to_string(blurred)
        
        return text.strip()  # Retire les espaces superflus et renvoie le texte détecté
    except Exception as e:
        print(f"Error in detect_and_read_text: {e}")
        return None

def main():
    try:  # Début du bloc try-except pour capturer les erreurs
        file_path = "imgbasler3.png"  # Chemin de l'image
        image = read_and_resize_image(file_path)  # Lit et redimensionne l'image
        circles = detect_circles(image)  # Détecte les cercles dans l'image
        canvas = draw_circles(image.copy(), circles)  # Dessine les cercles sur une copie de l'image
        center = get_average_circle(circles)  # Calcule le cercle moyen
        #canvas = draw_labels(canvas, center, center[2])  # Dessine les labels autour du cercle moyen
        canvas = draw_labels(canvas, center)  # Dessine les labels autour du cercle moyen
        polar, rotated = logpolar_and_rotate(image, center[:2])  # Applique la transformation logPolar et effectue une rotation
        if canvas is not None:  # Vérifie si le canevas est valide
            cv2.imshow("Canvas", canvas)  # Affiche le canevas
        if polar is not None:  # Vérifie si l'image transformée est valide
            cv2.imshow("polar", polar)  # Affiche l'image transformée
        if rotated is not None:  # Vérifie si l'image transformée est valide
            cv2.imshow("rotated", rotated)  # Affiche l'image transformée
        # Définir le chemin où l'image tournée sera enregistrée
        rotated_image_path = './rotated_image.jpg'
        # Enregistrer l'image tournée
        cv2.imwrite(rotated_image_path, rotated)
        preprocess_image(rotated_image_path)
        text = detect_and_read_text(rotated)  # Détecte et lit le texte dans l'image transformée
        if text:
            print("Detected Text:")
            print(text)
        cv2.waitKey()  # Attend une touche de clavier
        cv2.destroyAllWindows()  # Ferme toutes les fenêtres
        
    except Exception as e:  # Capture les erreurs potentielles
        print(f"Error in main: {e}")  # Affiche l'erreur s'il y en a une

if __name__ == "__main__":
    main()  # Appelle la fonction principale si le script est exécuté en tant que programme principal
    
#"""