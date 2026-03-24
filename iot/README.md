# VeinGuard Backend (Raspberry Pi)

Ce dossier contient l'API REST Flask et la base de données SQLite nécessaires pour faire fonctionner le système côté Raspberry Pi, ainsi que les scripts de traitement biométrique.

## Installation

1. S'assurer d'avoir Python 3.
2. Installer les dépendances :
   ```bash
   sudo apt-get install python3-opencv  # Recommandé pour Raspberry Pi
   pip install -r requirements.txt
   ```

## Lancement du Serveur

Pour initialiser la base de données et démarrer l'API :
```bash
python app.py
```
Le serveur écoutera sur `http://0.0.0.0:5000`.

## Endpoints de l'API

- `GET /api/status` : Vérifie que le serveur fonctionne.
- `POST /api/login` : Authentification via `username` et `password`.
- `GET /api/users` : Liste détaillée des utilisateurs enregistrés.
- `POST /api/users/enroll` : Enregistrer un nouvel utilisateur (biométrie optionnelle).
- `POST /api/scan` : Soumettre une image biométrique et valider le matching.
- `GET /api/logs` : Récupérer tout l'historique d'accès.

## Scripts Internes
- `app.py` : Serveur Flask gérant les requêtes de l'app Mobile.
- `pbbm.py` : Logique de l'algorithme "Personalized Best Bit Map" pour le scan veineux.
- `database.py` : Création et connexion de la structure SQLite.
- `biometrics_service.py` : Couche d'intégration entre l'API et la logique de traitement d'images.
