# Image Node.js pour le build de l'application
FROM node:18-alpine AS build

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers package.json et package-lock.json
COPY package*.json ./
RUN npm install

# Copier le reste des fichiers de l'application
COPY . .
RUN npm run build

# Image Nginx pour servir l'application
FROM nginx:alpine

# Copier le fichier de configuration Nginx personnalisé
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers statiques générés par React dans le répertoire de Nginx
COPY --from=build /app/build /usr/share/nginx/html

# Exposer le port 80 pour le serveur Nginx
EXPOSE 80

# Commande pour démarrer Nginx
CMD ["nginx", "-g", "daemon off;"] 

