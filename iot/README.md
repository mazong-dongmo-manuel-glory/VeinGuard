# BioGuard Access IoT

Ce dossier contient la passerelle Raspberry Pi du projet :

- acquisition biométrique paume / doigts
- lecture des capteurs de présence et de contact
- contrôle du LCD, des LEDs et du buzzer
- communication MQTT avec l'application mobile
- synchronisation Firebase pour les profils et l'historique
- cache local SQLite pour le mode edge / hors ligne

## Modules principaux

- [mqtt_gateway.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/mqtt_gateway.py) : boucle principale MQTT et orchestration
- [core/security_controller.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/core/security_controller.py) : coordination matériel
- [biometrics/biometrics_service.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/biometrics/biometrics_service.py) : extraction ORB + géométrie de la main
- [cloud/firebase_service.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/cloud/firebase_service.py) : intégration Firestore
- [database.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/database.py) : cache SQLite local
- [config.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/config.py) : configuration GPIO, MQTT, Firebase

## Capteurs et actionneurs

- caméra Raspberry Pi
- capteur tactile / contact sur GPIO
- capteur ultrasonique de proximité
- capteur PIR de mouvement
- LED verte
- LED rouge
- buzzer
- écran LCD I2C

## Topics MQTT

- `bioguard/cmd/auth/login`
- `bioguard/cmd/users/list`
- `bioguard/cmd/users/enroll`
- `bioguard/cmd/access/scan`
- `bioguard/cmd/access/logs`
- `bioguard/cmd/audit/list`
- `bioguard/cmd/settings/update`
- `bioguard/status`
- `bioguard/telemetry`
- `bioguard/events`

## Firebase

Le Pi peut fonctionner sans Firebase si :

- `VG_FIREBASE_ENABLED=0`
- ou si le fichier `firebase-service-account.json` n'est pas présent

Dans ce cas, le système garde :

- les utilisateurs dans SQLite
- les profils biométriques dans SQLite
- les événements d'accès dans SQLite

## Installation

```bash
cd iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

```bash
python mqtt_gateway.py
```

## Variables utiles

```bash
export VG_MOCK_MODE=1
export VG_MQTT_BROKER=localhost
export VG_FIREBASE_ENABLED=0
```
