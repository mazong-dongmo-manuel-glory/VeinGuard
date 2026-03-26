# BioGuard Access

Système IoT de contrôle d'accès biométrique multimodal basé sur Raspberry Pi, application mobile Expo, MQTT et Firebase.

Le projet a été réorienté autour du matériel réellement disponible :

- paume de la main
- géométrie des doigts
- capteur de lumière
- caméra Raspberry Pi

Les LEDs, le buzzer et l'écran LCD servent au retour utilisateur local. Les deux LED d’éclairage s’allument automatiquement quand la pièce devient sombre.

## Architecture

```text
Utilisateur
   |
   | paume / doigts
   v
Raspberry Pi
   |- Pi Camera -> extraction ORB + géométrie
   |- Light sensor -> détection de luminosité ambiante
   |- LCD + LEDs + buzzer -> feedback local
   |
   |- MQTT -> commandes et télémétrie temps réel
   |- SQLite -> cache edge et mode hors ligne
   |- Firebase -> profils, événements, historique, admin
   |- Firebase Authentication -> connexion mobile email / mot de passe
   v
Application mobile React Native (Expo)
```

## Dossiers

```text
VeinGuard/
├── Mobile/   # Application mobile Expo / React Native
├── iot/      # Passerelle Raspberry Pi, biométrie, capteurs, MQTT, Firebase
├── Maquettes/
└── docs/     # Documentation académique et projet
```

## Fonctionnalités

- enrôlement d'un utilisateur depuis l'application mobile
- modification et suppression d'utilisateurs via MQTT
- capture biométrique sur Raspberry Pi
- comparaison déterministe paume + doigts avec ORB + géométrie
- pilotage mobile du LCD, du buzzer, des LED et de l'éclairage
- historique d'accès et journaux d'audit côté mobile
- authentification Firebase email / mot de passe avec création de compte
- session persistante optionnelle et préférences par utilisateur
- listes mobiles en `FlatList` avec icônes
- synchronisation des profils et événements vers Firebase
- fonctionnement dégradé avec cache local SQLite si Internet tombe

## Stack

### IoT

- Python
- OpenCV
- gpiozero / RPi.GPIO
- paho-mqtt
- Firebase Admin SDK
- Picamera2

### Mobile

- React Native / Expo
- Zustand
- MQTT
- Firebase Web SDK

## Lancement

### IoT

```bash
cd iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mqtt_gateway.py
```

### Mobile

```bash
cd Mobile
npm install
npm start
```

Configurer ensuite les variables `EXPO_PUBLIC_FIREBASE_*` côté mobile et les variables `VG_FIREBASE_*` côté Raspberry Pi.

Pour la démonstration actuelle, les identifiants MQTT enregistrés dans le projet sont :

- utilisateur : `admin`
- mot de passe : `admin1234`

## Documents

- Documentation projet : [docs/Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md)
- README IoT : [iot/README.md](/Users/mazong/Documents/GitHub/VeinGuard/iot/README.md)
