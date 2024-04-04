import cv2  # Importation de la bibliothèque OpenCV pour le traitement d'images
import numpy as np  # Importation de la bibliothèque NumPy pour le traitement d'images

# 1. Conversion en niveaux de gris
def convert_to_grayscale(image):
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion de l'image en niveaux de gris
    except Exception as e:
        print("Une erreur s'est produite lors de la conversion en niveaux de gris:", e)

# 2. Réduction du bruit
def reduce_noise(image):
    try:
        return cv2.GaussianBlur(image, (5, 5), 0)  # Réduction du bruit en appliquant un flou gaussien
    except Exception as e:
        print("Une erreur s'est produite lors de la réduction du bruit:", e)


def ameliorer_contraste(image):
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_image)
    
    return clahe_image


# 3. Binarisation de l'image
def binarize_image(image):
    try:
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarisation de l'image
        return binary_image
    except Exception as e:
        print("Une erreur s'est produite lors de la binarisation de l'image:", e)


def binarize_image_adaptive(image):
    # Appliquer le seuillage adaptatif
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return binary_image



def preprocess_and_binarize(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur lors du chargement de l'image depuis {image_path}")
        return None
    
    # Prétraitement de l'image pour améliorer le contraste
    preprocessed_image = ameliorer_contraste(image)
    
    # Binarisation de l'image prétraitée
    binary_image = binarize_image_adaptive(preprocessed_image)
    
    return binary_image



# 4. Amélioration du contraste
def enhance_contrast(image):
    try:
        return cv2.equalizeHist(image)  # Amélioration du contraste de l'image en utilisant l'histogramme égalisé
    except Exception as e:
        print("Une erreur s'est produite lors de l'amélioration du contraste:", e)

# 5. Érosion et dilatation
def morphological_operations(image):
    try:
        kernel = np.ones((5, 5), np.uint8)  # Définition du noyau pour les opérations morphologiques
        erosion = cv2.erode(image, kernel, iterations=1)  # Érosion de l'image
        dilation = cv2.dilate(erosion, kernel, iterations=1)  # Dilatation de l'image
        return dilation
    except Exception as e:
        print("Une erreur s'est produite lors des opérations morphologiques:", e)

# 6. Extraction de zones de texte (détection de contours)
def extract_text_regions(image):
    try:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Détection des contours de l'image
        text_regions = []  # Initialisation de la liste des régions de texte
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # Obtenir les coordonnées du rectangle englobant
            text_regions.append((x, y, w, h))  # Ajout des coordonnées à la liste des régions de texte
        return text_regions
    except Exception as e:
        print("Une erreur s'est produite lors de l'extraction des zones de texte:", e)

# 7. Redimensionnement de l'image
def resize_image(image, width, height):
    try:
        return cv2.resize(image, (width, height))  # Redimensionnement de l'image à la taille spécifiée
    except Exception as e:
        print("Une erreur s'est produite lors du redimensionnement de l'image:", e)


###############################################################################################################

import cv2
import matplotlib.pyplot as plt
#from image_processing import convert_to_grayscale, reduce_noise, binarize_image, enhance_contrast, morphological_operations, extract_text_regions, resize_image

def preprocess_image(image_path):
    try:
        # Chargement de l'image
        image = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Affichage de l'image avec matplotlib
        plt.title('Image originale')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors du chargement de l'image:", e)
        return

    try:
        # Conversion de l'image en niveaux de gris
        gray_image = convert_to_grayscale(image)
        plt.imshow(gray_image, cmap='gray')  # Affichage de l'image en niveaux de gris avec matplotlib
        plt.title('Image en niveaux de gris')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors de la conversion en niveaux de gris:", e)
        return

    try:
        # Réduction du bruit de l'image
        blurred_image = reduce_noise(gray_image)
        plt.imshow(blurred_image, cmap='gray')  # Affichage de l'image avec réduction de bruit avec matplotlib
        plt.title('Image après réduction de bruit')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors de la réduction du bruit:", e)
        return

    try:
        # Binarisation de l'image
        binary_image = binarize_image(blurred_image)
        plt.imshow(binary_image, cmap='gray')  # Affichage de l'image binarisée avec matplotlib
        plt.title('Image binarisée')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors de la binarisation de l'image:", e)
        return

    try:
        # Amélioration du contraste de l'image
        enhanced_image = enhance_contrast(binary_image)
        plt.imshow(enhanced_image, cmap='gray')  # Affichage de l'image avec contraste amélioré avec matplotlib
        plt.title('Image avec contraste amélioré')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors de l'amélioration du contraste:", e)
        return

    try:
        # Application des opérations morphologiques à l'image
        processed_image = morphological_operations(enhanced_image)
        plt.imshow(processed_image, cmap='gray')  # Affichage de l'image après opérations morphologiques avec matplotlib
        plt.title('Image après opérations morphologiques')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors des opérations morphologiques:", e)
        return

    try:
        # Extraction des régions de texte de l'image
        text_regions = extract_text_regions(processed_image)
    except Exception as e:
        print("Une erreur s'est produite lors de l'extraction des zones de texte:", e)
        return

    try:
        # Redimensionnement de l'image à une taille spécifique
        final_image = resize_image(processed_image, 500, 300)
        plt.imshow(final_image, cmap='gray')  # Affichage de l'image finale avec matplotlib
        plt.title('Image finale')  # Titre de l'image
        plt.axis('off')  # Désactivation des axes
        plt.show()  # Affichage de l'image
    except Exception as e:
        print("Une erreur s'est produite lors du redimensionnement de l'image:", e)
        return

# Appel de la fonction pour traiter une image spécifique
#preprocess_image(r'C:\Users\thomas.ricou\test_image\texte_incurve.png')
