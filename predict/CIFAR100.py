import tensorflow as tf
import numpy as np
import io
from pathlib import Path
from utils.preprocessing import preprocess_image

class CIFAR100Predictor:
    def __init__(self):
        root_dir = Path(__file__).parent.parent
        model_path = root_dir / 'trained_models' / 'cifar100_model.h5'
        print(f"Tentative de chargement du modèle depuis : {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Impossible de trouver le modèle au chemin : {model_path}")

        self.model = tf.keras.models.load_model(model_path)

        self.cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm', 'truck'
        ]

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

    def predict_image(self, image_path):
        img_array = preprocess_image(image_path, target_size=(32, 32))
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.cifar100_classes[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, confidence

if __name__ == "__main__":
    predictor = CIFAR100Predictor()
