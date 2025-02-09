import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
import sys

class CIFAR10Predictor:
    def __init__(self, model_path='../models/cifar10_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.cifar10_classes = [
            'avion', 'automobile', 'oiseau', 'chat', 'cerf',
            'chien', 'grenouille', 'cheval', 'bateau', 'camion'
        ]

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

    def preprocess_image(self, image_path, target_size=(32, 32)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.cifar10_classes[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, confidence

if __name__ == "__main__":
    predictor = CIFAR10Predictor()
    predicted_class, confidence = predictor.predict_image('path_to_your_image.jpg')
    print(f"Je pense que cette image est : {predicted_class} avec une confiance de {confidence:.2f}")