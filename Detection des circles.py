import cv2
import numpy as np
import pytesseract

# Configuration du chemin d'accès à Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
img = cv2.imread("imgbasler3.png")
H, W = img.shape[:2]
img = cv2.resize(img, (W//2, H//2))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Définir les paramètres en fonction de la taille de l'image
minDist = max(H, W) // 50
param1 = 200
param2 = 100
minRadius = max(H, W) // 20
maxRadius = max(H, W) // 2

# Effectuer la détection de cercles avec les paramètres ajustés
circles = cv2.HoughCircles(blurred, method=cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, circles=None, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

"""
# Detect circles
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=100, 
    param1=100, 
    param2=30, 
    minRadius=50, 
    maxRadius=500
)"""

# Check if circles were found
if circles is not None:
    circles = np.intp(np.around(circles))
    
    # Assuming the first circle is the best one
    best_circle = circles[0][0]  
    x, y, r = best_circle
    
    # Calculate the mean center of the circle
    mean_center = (x, y)
    
    # Apply offsets if needed
    offset_x = 10
    offset_y = -5
    mean_center_adjusted = (mean_center[0] + offset_x, mean_center[1] + offset_y)
    
    # Draw the best circle and mean center on a copy of the original image
    canvas = img.copy()
    cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)
    cv2.circle(canvas, mean_center_adjusted, 5, (0, 0, 255), -1)

    # Appliquer un filtre de netteté à l'image en niveaux de gris
    sharp_img = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Extract the Region of Interest (ROI) based on the detected circle
    roi = img[y-r:y+r, x-r:x+r]

    # Preprocess the ROI (e.g., binarization, noise reduction, contrast enhancement)
    # Example: Convert to grayscale and apply thresholding
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, roi_thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply Log-Polar Transformation
    log_polar = cv2.logPolar(roi_gray, (int(r), int(r)), 100, cv2.WARP_FILL_OUTLIERS)

    # Rotate the transformed image 90 degrees counterclockwise
    log_polar_rotated = cv2.rotate(log_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Display the original ROI and the transformed image
    cv2.imshow("Original ROI", roi_gray)
    cv2.imshow("Log-Polar Transformed", log_polar_rotated)
    cv2.imshow("Best Circle with Mean Center", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Utilisation de Pytesseract pour extraire le texte de l'image prétraitée
    extracted_text = pytesseract.image_to_string(log_polar_rotated, lang='eng')

    # Affichage du texte extrait
    print("Texte extrait de l'image :")
    print(extracted_text)
    
else:
    print("No circles detected in the image.")
