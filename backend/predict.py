import tensorflow as tf
import numpy as np
from PIL import Image

# 1. PARAMETRES ET CLASSES
MODEL_PATH = 'model_enregistre.keras'
CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = (256, 256)

# 2. CHARGEMENT DU MODELE SAUVEGARDE
print("Chargement du modèle en cours...")
# Pour chager l'intégralité de l'architecture et des poids enregistré en moins d'une seconde, sans rien entrainer
model = tf.keras.models.load_model(MODEL_PATH)
print("modèle chargé avec succès !\n")


# 3. FONCTION DE PREDICTION SUR UNE IMAGE
def predict_leaf(image_input) -> tuple[str, float]:
    # Prend en entrée un chemin d'image (str) ou une image PIL

    # Si l'entrée est un chemin de fichier (chaine de caractère)
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    # Redimensionnement et conversion en tableau Numpy
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Inférence
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return predicted_class, confidence

# BLOC DE TEST LOCAL

# Ce bloc ne s'exécute que si on lance directement 'python predict.py'
# mais ne s'exécutera jamais si le fichier est importé d'ailleurs.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        classe, conf = predict_leaf(test_image_path)
        print(f"\n=== RESULTAT DU TEST ===")
        print(f"Image: {test_image_path}")
        print(f"CLASSE: {classe}")
        print(f"CONFIDENCE: {conf * 100:.2f}%\n")
    else:
        print("Usage: python predict.py <chemin_vers_image>")

