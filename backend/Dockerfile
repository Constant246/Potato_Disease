# Utiliser une image Python officielle
FROM python:3.11-slim

# définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances systèmes nécessaires à Pillow et OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les paquetages Python
RUN pip install --no-cache-dir  -r requirements.txt

# Copier tout le code du backend et le modèle entrainé
COPY . .

# Exposer le port 8000
EXPOSE 8000

# Commande de lancement du serveur Uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]


