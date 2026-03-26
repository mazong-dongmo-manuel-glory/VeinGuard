# Installation Et Execution

## Prerequis

### IoT

- Raspberry Pi OS
- Python 3
- caméra Pi configurée
- broker MQTT accessible
- GPIO disponibles

### Mobile

- Node.js
- npm
- Expo CLI via `npx expo`
- simulateur iOS ou appareil physique

## Installation IoT

```bash
cd iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Variables utiles IoT

```bash
export VG_MOCK_MODE=1
export VG_MQTT_BROKER=localhost
export VG_MQTT_PORT=1883
export VG_FIREBASE_ENABLED=0
```

Si Firebase est utilisé :

```bash
export VG_FIREBASE_ENABLED=1
export VG_FIREBASE_CREDENTIALS=/chemin/firebase-service-account.json
export VG_FIREBASE_PROJECT_ID=veinguard-d127f
```

## Execution IoT

```bash
cd iot
python mqtt_gateway.py
```

## Installation Mobile

```bash
cd Mobile
npm install
```

## Execution Mobile

```bash
cd Mobile
npm start
```

## Configuration Firebase mobile

La configuration Firebase côté mobile est centralisée dans :

- [Mobile/config.js](/Users/mazong/Documents/GitHub/VeinGuard/Mobile/config.js)

Produits attendus :

- Authentication
- Firestore

## Verification minimale

1. Démarrer le broker MQTT.
2. Lancer le backend IoT.
3. Lancer l'application mobile.
4. Vérifier la connexion Firebase.
5. Vérifier la réception de télémétrie.
6. Tester ajout, modification et suppression d'utilisateur.
7. Tester un scan d'accès.
