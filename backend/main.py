# This is a sample Python script.
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Les images du Dataset ont une taille de 256 x 256
IMAGE_SIZE = 256
# Pour définir le nombre d'images traitées simultanément par le réseau lors de l'entrainement
BATCH_SIZE = 32
CHANNELS = 3              # Rouge, Vert et Bleu (RGB)
N_CLASSES = 3             # Early Blight, Late Blight and Healthy

# 1. Chargement du dossier d'entrainement
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/train',
    shuffle = True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# 2. Chargement du dossier de validation initial
# Chargement et mise en batch sur le dossier 'Dataset/val'
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/val',
    shuffle = True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Séparation du dossier 'val' en Validation (50%) et en Test (50%)
# Pour compter le nombre de batchs contenus dans le jeu de 'validation brut'
val_batches = int(tf.data.experimental.cardinality(val_ds_raw).numpy())
# Pour prendre la moitié des batchs pour constituer le 'jeu de validation final'
val_ds = val_ds_raw.take(val_batches // 2)
# Prendre la moitié restante pour constituer le 'jeu de test indépendant'
test_ds = val_ds_raw.skip(val_batches // 2)

print(f"Batches d'entrainenment : {tf.data.experimental.cardinality(train_ds)}")
print(f"Batches de Validation : {tf.data.experimental.cardinality(val_ds)}")
print(f"Batches de Test : {tf.data.experimental.cardinality(test_ds)}")

# ================================================================================
# Optimisation des performances du PIPELINE (Cache & Prefetch)
# ================================================================================
"""
Sélection automatique du nombre optimal de threads processeur (sous tache d'exécution gérée 
indépendamment par notre processeur) en fonction de notre CPU
"""
AUTOTUNE = tf.data.AUTOTUNE
# .cache() : Garde les images en mémoire RAM après la première époque
# .shuffle(1000) : Mélange un ensemble de 1000 images à chaque époque pour casser tout ordre d'apprentissage
# .prefetch() : Prépare les futurs batchs en mémoire pendant que le GPU calcule le batch en cours
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# Pas de shuffle ici pour garantir les évaluations reproductibles
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ====================================================================================
# PRE-TRAITEMENT ET DATA AUGMENTATION
# ====================================================================================

# On va créer un modèle séquentiel Keras contenant les couches de redimensionnement et de normalisation
resize_and_rescale = tf.keras.Sequential([
    # Redimensionne de manière explicite les images aux dimension (256 x 256) pixels
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),

    # Divise la valeur de chaque pixel par 255.0 pour convertir la plage [0, 255] vers [0.0, 1.0]
    # c'est la normalisation
    layers.Rescaling(1.0 / 255)
])

# Créer un modèle séquentiel Keras dédié aux transformations aléatoire d'images
data_augmentation = tf.keras.Sequential([
    # Applique aléatoirement un retournement miroir horizontal, vertical ou les deux
    layers.RandomFlip("horizontal_and_vertical"),

    # Applique une rotation aléatoire dans une plage de 20% autour du centre
    layers.RandomRotation(0.2),
])

# =====================================================
# 6. CONSTRUCTION DU MODELE CNN (PHASE 3)
# =====================================================

# Pour spécifier les dimensions exactes d'entrées
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
# Pour initialiser le modèle séquentiel dans lequel nous empilons les couches une par une
model = models.Sequential([
    # Pour intégrer les règles de Prétraitement au sein du modèle pour exécution automatique
    # pendant les entrainement et prédiction
    resize_and_rescale,
    data_augmentation,

    # Bloc de Convolution 1
    # première couche : applique 32 filtres de taille 3X3 pixels avec la fonction d'activation RELU
    # pour extraire les contours et couleurs primaires.
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    # Diviser par 2 hauteur et largeur pour réduire la taille en conservant les infos dominantes.
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Bloc Convolutif 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Bloc Convolutif 3
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Bloc Convolutif 4
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Transition et réseau Dense
    # Pour transformer les cartes 3D réduites en un simple vecteur 1D (liste de nombre)
    layers.Flatten(),
    # Couche entièrement connectée de 64 neurones pour combiner l'ensemble des motifs extraits
    # pour raisonner sur la maladie
    layers.Dense(64, activation="relu"),
    layers.Dense(N_CLASSES, activation="softmax")
])

# Construire explicitement les dimensions du réseau
# Force l'inialisation des poids du modèle
model.build(input_shape=input_shape)

# Résumé de l'architecture
model.summary()

# ================================================================
# ETAPE 7 : COMPILATION DU MODELE
# ================================================================

model.compile(
    # Standard en DL pour gérer automatiquement le taux d'apprentissage pour chaque paramètre du réseau
    optimizer = 'Adam',
    # Pour calculer l'écart entre la maladie prédite et la vraie maladie de l'image
    # from_logits = False : Précise que notre couche de sortie applique déjà la focntion Softmax
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    # Pour suivre la précision du modèle
    metrics = ['accuracy']
)

# ==================================================================
# ETAPE 8 : ENTRAINEMENT DU MODELE
# ==================================================================
# Fixe le nombre de passages complet du réseau sur les 1721 images d'entrainement
EPOCHS = 20

print("\n --- Début de l'entrainement du Modèle ---")
# Pour déclencher la boucle d'entrainement
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    # Evalue la précision du réseau à la fin de chaque époque sur des images qu'il n'a
    # utilisé pour s'entrainer, cela permet de savoir s'il y a du surapprentissage
    validation_data=val_ds,
    # Affiche une barre de progression en temps réel pour chaque époque.
    verbose=1
)

# =========================================================
# EVALUATION DU MODEL
# =========================================================
print("\nEvaluation sur le jeu de Test (Données jamais vu !)")
# Pour faire passer les images de test à travers le réseau entrainé
test_score = model.evaluate(test_ds)
# Affiche ma valeur de perte avec 4 chiffre après la virgule
print(f"Perte sur le jeu de Test : {test_score[0]:.4f}")
# La valeur de précision multiplié par 100 avec 2 chiffre après la virgule
print(f"Précision sur le jeu de Test : {test_score[1]*100:.2f}%")

# =============================================================
# SAUVEGARDE DU MODELE : On veut éviter de réentrainer le modèle à chaque exécution
# =================================================================================
model.save('model_enregistre.keras')
print("Modèle sauvegardé sous le nom de : 'model_enregistre.keras' !")

# ===============================================================
# VISUALISATION DES COURBES
# ===============================================================
# Pour extraire le Journal d'Historique dans l'objet 'history' suite à la commande 'model.fit'
# Précision d'Entrainement
acc = history.history['accuracy']
# Précision de validation
val_acc = history.history['val_accuracy']
# Perte d'Entrainement
loss = history.history['loss']
# Perte de Validation
val_loss = history.history['val_loss']
# 12 pouces de largeur et 5 pouces de Hauteur pour la taille du graphique
plt.figure(figsize=(12, 5))

# Graphique 1 : Précision (Accuracy)
# Divise la fénètre en une grille de 1 ligne et 2 colonnes, puis sélectionne la première case à gauche
# pour tracer le premier graphique.
plt.subplot(1, 2, 1)
# Pour tracer la courbe de précision d'entrainement en Bleu et celle de la validation en Orange
# au fil des 20 époques (de 0 à 19)
plt.plot(range(EPOCHS), acc, label='Précision Entrainement')
plt.plot(range(EPOCHS), val_acc, label='Précision Validation')
# Configure les détails du premier graphique
plt.legend(loc='lower right')
plt.title('Précision au fil des époques')
plt.xlabel('Epoques')
plt.ylabel('Précision')

# Graphique 2 : Perte (loss)
# Sélectionne la 2 case de la grille (à droite) pour dessiner le deuxième graphique
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Perte Entrainement')
plt.plot(range(EPOCHS), val_loss, label='Perte Validation')
plt.legend(loc='upper right')
plt.title('Perte au fil des époques')
plt.xlabel('Epoques')
plt.ylabel('Perte')
# Je commente cette ligne pour permettre l'affichage dans le navigateur
# plt.show()


"""
for image_batch, label_batch in train_ds.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.title(train_ds.class_names[label_batch[0]])
    plt.axis('off')
    plt.show()
"""

