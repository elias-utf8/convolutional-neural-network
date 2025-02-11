import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, target_size=(32, 32)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
