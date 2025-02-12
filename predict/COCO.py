import tensorflow as tf
import io
from pathlib import Path
from utils.preprocessing import preprocess_image  

class COCOPredictor:
    def __init__(self):
        root_dir = Path(__file__).parent.parent
        model_path = root_dir / 'trained_models' / 'coco_model.h5'
        print(f"Tentative de chargement du modèle depuis : {model_path}")

        try:
            self.model = tf.keras.models.load_model(str(model_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Impossible de trouver le modèle au chemin : {model_path}")

        # Définition des classes COCO
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
            "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
            "hair drier", "toothbrush"
        ]

        # Vérifier que le modèle et la liste de classes correspondent
        if len(self.class_names) != self.model.output_shape[-1]:
            print(f"⚠️ Attention : {len(self.class_names)} noms de classes définis, "
                  f"mais le modèle attend {self.model.output_shape[-1]} sorties.")

    def predict_image(self, image_path):
        img_array = preprocess_image(image_path, target_size=(32, 32))
        predictions = self.model.predict(img_array)
        predicted_class_index = tf.argmax(predictions[0]).numpy()
        predicted_class = self.class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        return predicted_class, confidence

    def summary(self):
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

if __name__ == "__main__":
    predictor = COCOPredictor()
    predicted_class, confidence = predictor.predict_image('path_to_your_image.jpg')
    print(f"Image prédite : {predicted_class}, confiance : {confidence:.2f}")