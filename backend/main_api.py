import io
import numpy as np
import tensorflow as tf
# Ce Module est utilisé en python pour lire des images
from PIL import Image
# Structure principale de FastAPI nécessaire à la réception de fichiers.
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# Pour exécuter l'application FastAPI
import uvicorn

# 1. INITIALISATION DE L'APPLICATION ET DU MODELE
app = FastAPI(title="API de Détection des Maladies de la pomme de terre")

# Configuration CORS pour autoriser React à communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du Modèle au démarrage de l'API
MODELE = tf.keras.models.load_model('model_enregistre.keras')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
IMAGE_SIZE = 256

# 2. FONCTION UTILITAIRE : Convertir une image reçu en un Tableau Numpy
def read_file_as_img(data) -> np.ndarray:
    # Lire les octets bruts  et les convertir en image PIL
    image = Image.open(io.BytesIO(data)).convert('RGB')
    # Redimensionner l'image à la taille attendue par le modèle
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Convertir l'image PIL en Tableau Numpy
    return np.array(image)

# ROUTE RACINE (Test de bon fonctionnement)
@app.get("/ping")
async def ping():
    return {"Status": "l'API fonctionne parfaitement !"}

# ROUTE DE PREDICTION
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Pour lire le fichier envoyé par le frontend
    bytes_data = await file.read()

    # Convertir les octets en tableau Numpy
    image = read_file_as_img(bytes_data)

    # Ajouter la dimension de Batch -> (1, 256, 256, 3)
    img_batch = np.expand_dims(image, axis=0)

    # Prédiction avec le modèle
    predictions = MODELE.predict(img_batch)

    # Extraction des classes et la Confiance
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]) * 100)

    # Renvoyer la réponse au format JSON
    return {
        "filename": file.filename,
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

# 5. POINT D'ENTREE POUR LANCER LE SERVEUR
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


