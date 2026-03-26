# BioGuard Access IoT

Ce dossier contient la passerelle Raspberry Pi du projet :

- acquisition biométrique paume / doigts
- lecture du capteur de lumière
- contrôle du LCD, du buzzer, de la LED rouge, de la LED verte et des deux LED d'éclairage
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
- capteur de lumière sur GPIO
- LED verte
- LED rouge
- LED d'éclairage 1
- LED d'éclairage 2
- buzzer
- écran LCD I2C

## Pilotage mobile

L'application mobile échange uniquement via MQTT et Firebase :

- récupération de la télémétrie sur `bioguard/telemetry`
- envoi des commandes matérielles sur `bioguard/cmd/settings/update`
- accusé de réception sur `bioguard/res/settings/update/<client_id>`

Le mobile peut :

- lire l'état du capteur de lumière
- voir l'état de la caméra, du buzzer, du LCD et des LED
- activer ou désactiver le mode automatique des LED d'éclairage
- forcer l'allumage des LED d'éclairage
- piloter la LED rouge et la LED verte
- déclencher un test buzzer
- envoyer un message au LCD

## Topics MQTT

- `bioguard/cmd/auth/login`
- `bioguard/cmd/users/list`
- `bioguard/cmd/users/enroll`
- `bioguard/cmd/users/update`
- `bioguard/cmd/users/delete`
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
export VG_MQTT_USERNAME=admin
export VG_MQTT_PASSWORD=admin1234
export VG_FIREBASE_ENABLED=0
```

Identifiants MQTT de démonstration enregistrés dans le projet :

- utilisateur : `admin`
- mot de passe : `admin1234`

Commandes Raspberry Pi recommandées :

```bash
sudo mosquitto_passwd -b /etc/mosquitto/passwd admin admin1234
sudo systemctl restart mosquitto

mosquitto_pub -h localhost -p 1883 -u admin -P admin1234 -t test -m hello

cd ~/Desktop/VeinGuard/iot
source .venv/bin/activate
export VG_MQTT_BROKER=localhost
export VG_MQTT_PORT=1883
export VG_MQTT_USERNAME=admin
export VG_MQTT_PASSWORD=admin1234
python mqtt_gateway.py
```
