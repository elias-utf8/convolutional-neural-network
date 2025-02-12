"""
Réseau de neurones convolutifs - modèle rudimentaire de classification d'image
utilisant le dataset COCO
Auteur : Elias GAUTHIER
Date : 02/02/2025
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

dataset, info = tfds.load('coco/2017', 
                         split='train',
                         shuffle_files=True,
                         with_info=True)

num_classes = info.features['objects']['label'].num_classes
class_names = info.features['objects']['label'].names

def prepare_data(example):
    image = tf.cast(example['image'], tf.float32) / 255.0
    image = tf.image.resize(image, (32, 32))
    if tf.size(example['objects']['label']) > 0:
        label = tf.one_hot(example['objects']['label'][0], num_classes)
    else:
        label = tf.one_hot(0, num_classes)
    return image, label

train_ds = dataset.map(prepare_data).batch(32).prefetch(tf.data.AUTOTUNE)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

train_size = int(0.8 * len(list(train_ds)))
val_ds = train_ds.skip(train_size)
train_ds = train_ds.take(train_size)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping]
)

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

model.save('trained_models/coco_model.h5')