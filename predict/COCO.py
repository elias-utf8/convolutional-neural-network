import tensorflow as tf
import io
from pathlib import Path

from utils.preprocessing import preprocess_image  

class COCOPredictor:
    def __init__(self):
        root_dir = Path(__file__).parent.parent
        model_path = root_dir / 'trained_models' / 'COCO_model.h5'
        print(f"Tentative de chargement du modèle depuis : {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Impossible de trouver le modèle au chemin : {model_path}")

        self.model = tf.keras.models.load_model(model_path)

    def predict_image(self, image_path):
        img_array = preprocess_image(image_path, target_size=(32, 32))
        predictions = self.model.predict(img_array)
        predicted_class_index = tf.argmax(predictions[0]).numpy()
        predicted_class_name = self.class_names[predicted_class_index]
        return predicted_class_name

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

if __name__ == "__main__":
    predictor = COCOPredictor()
