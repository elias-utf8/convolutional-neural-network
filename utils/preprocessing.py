import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, target_size=(32, 32)):
    """
    Charge et prétraite une image pour la prédiction avec un CNN.

    Args:
        image_path (str): Chemin de l'image à charger.
        target_size (tuple): Taille cible de l'image.

    Returns:
        np.array: Image normalisée et prête pour la prédiction.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
