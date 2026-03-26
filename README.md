# BioGuard Access

Système IoT de contrôle d'accès biométrique multimodal basé sur Raspberry Pi, application mobile Expo, MQTT et Firebase.

Le projet a été réorienté pour reconnaître une personne avec plusieurs sources biométriques et contextuelles :

- paume de la main
- géométrie des doigts
- capteur de proximité
- capteur de contact / toucher
- capteur de mouvement

Les LEDs, le buzzer et l'écran LCD servent au retour utilisateur en temps réel pendant l'enrôlement et la décision d'accès.

## Architecture

```text
Utilisateur
   |
   | paume / doigts
   v
Raspberry Pi
   |- Pi Camera -> extraction ORB + géométrie
   |- Touch sensor -> validation de contact
   |- Ultrasonic sensor -> présence devant le lecteur
   |- PIR / motion -> activité autour de la porte
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
- capture biométrique sur Raspberry Pi
- comparaison déterministe paume + doigts avec ORB + géométrie
- feedback local par LCD, LED verte, LED rouge et buzzer
- historique d'accès côté mobile
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

## Documents

- Documentation projet : [docs/Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md)
- README IoT : [iot/README.md](/Users/mazong/Documents/GitHub/VeinGuard/iot/README.md)
