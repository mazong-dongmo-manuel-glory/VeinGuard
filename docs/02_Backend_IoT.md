# Backend IoT

## Objectif

Le backend IoT orchestre le matériel connecté, le pipeline biométrique, la persistance locale et la communication avec le mobile.

## Modules

- [config.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/config.py) : constantes, topics MQTT, GPIO, paramètres Firebase
- [mqtt_gateway.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/mqtt_gateway.py) : point d'entrée principal et gestionnaire MQTT
- [core/security_controller.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/core/security_controller.py) : logique métier temps réel
- [biometrics/biometrics_service.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/biometrics/biometrics_service.py) : extraction et vérification biométriques
- [database.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/database.py) : SQLite local
- [cloud/firebase_service.py](/Users/mazong/Documents/GitHub/VeinGuard/iot/cloud/firebase_service.py) : synchronisation Firestore
- [hardware/](/Users/mazong/Documents/GitHub/VeinGuard/iot/hardware) : drivers matériels

## Matériel effectivement géré

### Capteurs

- caméra Raspberry Pi
- capteur de lumière sur GPIO

### Actionneurs

- LED verte
- LED rouge
- deux LED d'éclairage
- buzzer
- écran LCD I2C 16x2

## Contrôleur matériel

Le contrôleur :

- exécute une séquence de démarrage
- synchronise l'éclairage selon la luminosité
- affiche l'état sur LCD
- donne le retour visuel et sonore après décision
- produit une télémétrie structurée

## Pipeline biométrique

Le backend ne repose pas sur un gros modèle IA. Il utilise une approche déterministe :

- segmentation de la main dans une ROI
- extraction de caractéristiques géométriques
- moments de Hu
- points ORB et signature de texture
- comparaison pondérée contre un profil enregistré

Avantages :

- léger sur Raspberry Pi
- explicable pendant la démonstration
- facile à calibrer

## Persistance

### SQLite

SQLite sert de cache edge pour :

- utilisateurs
- profils biométriques
- événements d'accès
- journaux d'audit
- état du dispositif

### Firebase

Firebase sert au stockage distant pour :

- profils utilisateurs
- profils biométriques
- événements d'accès
- télémétrie

Si Firebase est indisponible, le système continue avec SQLite local.

## Commandes MQTT prises en charge

- authentification de secours MQTT
- liste des utilisateurs
- enrôlement
- mise à jour utilisateur
- suppression utilisateur
- scan d'accès
- lecture des logs d'accès
- lecture des logs d'audit
- mise à jour des réglages du système

## Gestion d'erreurs

Le backend gère :

- JSON invalide
- identifiants manquants
- utilisateur absent
- profil biométrique absent
- erreur de capture
- erreur de seuil sur le capteur de lumière
- mode mock si matériel ou Firebase indisponibles

## Limites connues

- le projet ne dispose pas de 5 capteurs matériels distincts
- la qualité biométrique dépend de la lumière et du cadrage
- la caméra et le capteur de lumière sont les seuls capteurs physiques réellement intégrés aujourd'hui
