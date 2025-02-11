import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import io
import sys
from pathlib import Path


class MobileNetPredictor:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
        
    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict_image(self, image_path, top_predictions=1):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        
        decoded_predictions = decode_predictions(predictions, top=top_predictions)[0]
        
        predicted_class = decoded_predictions[0][1]
        confidence = decoded_predictions[0][2]
        
        return predicted_class, confidence
    
    def get_top_predictions(self, image_path, top_k=3):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        return decode_predictions(predictions, top=top_k)[0]

if __name__ == "__main__":
    predictor = MobileNetPredictor()