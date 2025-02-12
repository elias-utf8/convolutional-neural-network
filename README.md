![Version](https://img.shields.io/github/v/tag/elias-utf8/convolutional-neural-network?label=version&color=blue)
# Réseau neuronal convolutif de reconnaissance d'images

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/TensorFlow_logo.svg" width="350" style="vertical-align: middle" />
</div>

<br>

*La documentation du projet est encore en cours. Ceci n'est qu'un survol du vaste monde du ML et des CNN. Toutes ces informations sont disponibles sur Internet.*

![Aperçu de l'application](screenshots/app_screen.png)

## Introduction aux CNN

Les réseaux de neurones convolutifs (CNN) sont une classe de réseaux de neurones artificiels principalement utilisés pour l'analyse d'images. Ils sont inspirés par le cortex visuel des animaux et sont particulièrement efficaces pour des tâches telles que la reconnaissance d'images, la classification et la détection d'objets.

Un CNN traite une image sous forme de matrice de pixels :

| ![Tux_2](screenshots/tux_1.png)  | ![Tux_3](screenshots/tux_2.png)  |
|----------------------------------|----------------------------------|
| Image sous forme classique       | Image traitée par le CNN         |

### Architecture de base

Un CNN est composé de plusieurs types de couches :

1. **Couches de convolution** : Appliquent des filtres (ou noyaux) pour extraire des caractéristiques locales de l'image. Chaque filtre détecte des motifs spécifiques comme les bords, les textures, etc.
2. **Couches de pooling** : Réduisent la dimensionnalité des données tout en conservant les informations importantes. Le pooling max est couramment utilisé, où la valeur maximale d'une région est conservée.
3. **Couches Fully Connected (FC)** : Après plusieurs couches de convolution et de pooling, les données sont aplaties et passées à travers des couches entièrement connectées pour la classification finale.

Voici un exemple de filtres (noyaux ou *kernels* en anglais) classiques pouvant être utilisés par un CNN :

| ![Tux_2](screenshots/tux_3.png)                                                | ![Tux_3](screenshots/tux_4.png)                                                  |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `[-10,0,10],[-10,0,10],[-10,0,10]` : Mise en évidence des traits **verticaux**.| `[10,10,10],[0,0,0],[-10,-10,-10]` : Mise en évidence des traits **horizontaux**.|

Exemple pour appliquer une matrice de filtre sur une image en Python :
```py
'''
Applique un filtre mettant en évidence les traits verticaux
'''
def filtre_2(image_nb):
    kernel = np.matrix([[-10,0,10],[-10,0,10],[-10,0,10]])
    print(kernel)
    img_1 = cv2.filter2D(image_nb, -1, kernel)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    plt.show()
```
Il existe un fabuleux site expliquant cela de manière interactive : [Setosa.io](https://setosa.io/ev/image-kernels/)

> **C'est ici que réside tout l'intérêt des CNN : les valeurs des filtres sont apprises pendant l'entraînement. Le réseau ajuste automatiquement ces valeurs pour extraire les caractéristiques les plus pertinentes pour la tâche donnée.**

### Fonctionnement

1. **Entrée** : Une image est fournie en entrée sous forme de matrice de pixels.
2. **Convolution** : Des filtres glissent sur l'image pour créer des cartes de caractéristiques.
3. **Pooling** : Réduit la taille des cartes de caractéristiques tout en conservant les informations importantes.
4. **Flattening** : Convertit les cartes de caractéristiques en un vecteur unidimensionnel.
5. **Fully Connected Layers** : Effectuent la classification finale en utilisant les caractéristiques extraites.
6. **Sortie** : Produit une probabilité pour chaque classe possible.

> _Je ne détaillerai pas chacune de ces étapes, car cela relève de connaissances que je n'ai pas encore entièrement acquises._

### Entraînement

Afin d'obtenir un modèle de CNN fonctionnel, il est nécessaire d'entraîner ce dernier sur de grands jeux de données nommés *datasets*. On peut trouver des *datasets* sur Internet, notamment sur [Kaggle](https://www.kaggle.com/datasets).

Les programmes d'entraînement des modèles se trouvent dans `/models`, et les modèles entraînés dans `/trained_models`.

Vous pouvez constituer votre propre *dataset*, mais cela est une tâche assez longue. Mon application n'est pas destinée à cela. Vous devrez également créer des répertoires de validation/entraînement et modifier la structure des programmes d'entraînement que j'ai écrits.

N'oubliez pas que **plus de données = meilleure efficacité de prédiction**.
Grâce à TensorFlow, on peut accéder à des *datasets* directement via l'importation de la bibliothèque.

Exemple pour le dataset [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) : 
```py
from tensorflow.keras.datasets import cifar10
```

---
## Implémentation
Dans mon cas, j'ai entraîné trois modèles différents que j'ai nommés d'après le *dataset* sur lequel ils ont été entraînés :
**CIFAR10**, **CIFAR100** et **COCO**.

Pour construire ces CNN, j'utiliserai l'un des outils les plus réputés et utilisés dans le domaine, à savoir **TensorFlow**. C'est un outil d'auto-apprentissage **open-source** développé par Google.

