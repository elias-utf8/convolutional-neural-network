import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Charger le modèle pré-entraîné
model = MobileNetV2(weights="imagenet")

# Fonction pour charger et afficher une image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Redimensionner pour MobileNet
        display_image(image)
        classify_image(image)

# Fonction pour afficher l'image dans l'interface
def display_image(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image  # Garder une référence pour éviter la suppression par le garbage collector

# Fonction pour classer l'image
def classify_image(image):
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    image = preprocess_input(image)  # Prétraiter l'image pour MobileNet
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 prédictions
    result_text.set("\n".join([f"{label}: {prob:.2f}%" for (_, label, prob) in decoded_predictions]))

# Créer la fenêtre principale
root = tk.Tk()
root.title("Reconnaissance d'Objets")

# Bouton pour charger une image
load_button = tk.Button(root, text="Charger une image", command=load_image)
load_button.pack(pady=10)

# Panneau pour afficher l'image
panel = tk.Label(root)
panel.pack()

# Zone de texte pour afficher les résultats
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12))
result_label.pack(pady=10)

# Lancer l'interface
root.mainloop()