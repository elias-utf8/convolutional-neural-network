import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('models/cifar100_model.h5')
model.summary();
cifar10_classes = [
    'avion', 'automobile', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

def preprocess_image(image_path, target_size=(32, 32)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = cifar10_classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

image_path = 'tests/avion.png'

predicted_class, confidence = predict_image(image_path)
print(f"Je pense que cette image est : {predicted_class} avec une confiance de {confidence:.2f}")