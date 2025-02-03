import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

model = MobileNetV2(weights="imagenet")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        display_image(image)
        classify_image(image)

def display_image(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image 

def classify_image(image):
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    result_text.set("\n".join([f"{label}: {prob:.2f}%" for (_, label, prob) in decoded_predictions]))

root = tk.Tk()
root.title("Reconnaissance d'Objets")

load_button = tk.Button(root, text="Charger une image", command=load_image)
load_button.pack(pady=10)


panel = tk.Label(root)
panel.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12))
result_label.pack(pady=10)

# Lancer l'interface
root.mainloop()