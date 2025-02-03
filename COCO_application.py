import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

dataset, info = tfds.load('coco/2017', split='train', with_info=True)
class_names = info.features['objects']['label'].names

model = load_model('models/coco_model.h5')

def predict_image(image_path, model, class_names):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (32, 32))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0) 

    # Faire la prédiction
    predictions = model.predict(image)
    predicted_class_index = tf.argmax(predictions[0]).numpy()  
    predicted_class_name = class_names[predicted_class_index]  

    return predicted_class_name

image_path = 'tests/burger.jpg'  
predicted_class = predict_image(image_path, model, class_names)
print(f"La classe prédite est : {predicted_class}")

image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
plt.imshow(image)
plt.title(f"Prédiction : {predicted_class}")
plt.axis('off')
plt.show()