"""
Réseau de neurones convolutifs pour ImageNet Mini
Auteur : Elias GAUTHIER
Date : 02/02/2025
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paramètres
image_size = (224, 224) 
num_classes = 100  
batch_size = 32 
epochs = 50  

# Chemins des dossiers
train_dir = 'train'  
val_dir = 'valid' 

# Prétraitement des données avec ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalisation des pixels entre 0 et 1
    rotation_range=20,  # Augmentation des données : rotation
    width_shift_range=0.2,  # Augmentation des données : décalage horizontal
    height_shift_range=0.2,  # Augmentation des données : décalage vertical
    shear_range=0.2,  # Augmentation des données : cisaillement
    zoom_range=0.2,  # Augmentation des données : zoom
    horizontal_flip=True,  # Augmentation des données : retournement horizontal
    validation_split=0.2  # Fraction des données réservée à la validation
)

# Chargement des données d'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Pour un problème de classification multi-classes
    subset='training'  # Données d'entraînement
)

# Chargement des données de validation
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Données de validation
)

# Modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 100 classes pour ImageNet Mini
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping pour éviter le surapprentissage
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'\nPrécision sur le jeu de validation : {test_accuracy:.4f}')

# Visualisation des résultats
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision (entraînement)')
plt.plot(history.history['val_accuracy'], label='Précision (validation)')
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte (entraînement)')
plt.plot(history.history['val_loss'], label='Perte (validation)')
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()

# Sauvegarder le modèle
model.save('imagenet_mini_model.h5')