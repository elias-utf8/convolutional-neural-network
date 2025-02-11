import tensorflow as tf
import numpy as np
import io
from utils.preprocessing import preprocess_image
from pathlib import Path

class CIFAR10Predictor:
    def __init__(self):
        root_dir = Path(__file__).parent.parent
        model_path = root_dir / 'trained_models' / 'cifar10_model.h5'
        print(f"Tentative de chargement du modèle depuis : {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Impossible de trouver le modèle au chemin : {model_path}")
            
        self.cifar10_classes = [
            'avion', 'automobile', 'oiseau', 'chat', 'cerf',
            'chien', 'grenouille', 'cheval', 'bateau', 'camion'
        ]

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

    def predict_image(self, image_path):
        img_array = preprocess_image(image_path, target_size=(32, 32))
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.cifar10_classes[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, confidence

if __name__ == "__main__":
    predictor = CIFAR10Predictor()
