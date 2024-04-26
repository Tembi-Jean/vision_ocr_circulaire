import numpy as np
from skimage import color, exposure, filters
from scipy.stats import entropy

# Luminance
def luminance(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Calcul de la luminance moyenne de l'image
    luminance_value = np.mean(gray_image)
    return luminance_value

# Écart type des niveaux de gris
def gray_std_deviation(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Calcul de l'écart type des niveaux de gris de l'image
    std_deviation_value = np.std(gray_image)
    return std_deviation_value

# Niveau de gris moyen
def gray_mean(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Calcul du niveau de gris moyen de l'image
    mean_value = np.mean(gray_image)
    return mean_value

# Saturation
def saturation(image):
    # Conversion de l'image en espace de couleur HSV
    hsv_image = color.rgb2hsv(image)
    # Calcul de la saturation moyenne de l'image
    saturation_value = np.mean(hsv_image[:,:,1])
    return saturation_value

# Entropie
def image_entropy(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Calcul de l'entropie de l'image
    entropy_value = entropy(gray_image)
    return entropy_value

# Sharpness (Netteté)
def sharpness(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Application du filtre de Sobel pour détecter les contours
    edges = filters.sobel(gray_image)
    # Calcul de la netteté de l'image en utilisant la norme L2 des gradients
    sharpness_value = np.sqrt(np.mean(edges**2))
    return sharpness_value

# Bruit
def noise(image):
    # Conversion de l'image en niveau de gris
    gray_image = color.rgb2gray(image)
    # Calcul du rapport signal-bruit (PSNR)
    noise_value = np.var(gray_image)
    return noise_value