import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import io

class COCOPredictor:
    def __init__(self, model_path='../models/coco_model.h5'):
        self.model = tf.keras.models.load_model(model_path)

    def predict_image(self, image_path, model, class_names):

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (32, 32))
        image = image / 255.0
        image = tf.expand_dims(image, axis=0) 

        predictions = model.predict(image)
        predicted_class_index = tf.argmax(predictions[0]).numpy()  
        predicted_class_name = class_names[predicted_class_index]  

        return predicted_class_name

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()


if __name__ == "__main__":
    predictor = COCOPredictor()
    predicted_class, confidence = predictor.predict_image('path_to_your_image.jpg')

