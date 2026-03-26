# Architecture Globale

## Vue d'ensemble

BioGuard Access est un système de contrôle d'accès connecté composé de deux blocs :

- un backend IoT exécuté sur Raspberry Pi
- une application mobile Expo / React Native

Le Raspberry Pi effectue la capture biométrique, le pilotage matériel local et la télémétrie. L'application mobile sert à administrer les utilisateurs, lancer des actions et consulter les événements.

## Architecture logique

```text
Application mobile
  |- Auth Firebase
  |- Préférences utilisateur
  |- MQTT client
  |- Écrans administration / historique / paramètres
          |
          | MQTT
          v
Passerelle Raspberry Pi
  |- MQTT gateway
  |- Contrôleur matériel
  |- Biométrie paume + géométrie des doigts
  |- SQLite local
  |- Synchronisation Firebase
          |
          +- Caméra
          +- Capteur de lumière
          +- LCD I2C
          +- Buzzer
          +- LED rouge / verte
          +- 2 LED d'éclairage
```

## Flux principaux

### Enrôlement

1. L'administrateur crée ou complète un profil dans l'application mobile.
2. Le mobile publie une commande MQTT d'enrôlement.
3. Le Raspberry Pi guide l'utilisateur via le LCD.
4. La caméra capture la main.
5. Un profil biométrique déterministe est généré.
6. Le profil est enregistré en SQLite puis synchronisé vers Firebase.

### Contrôle d'accès

1. Une commande de scan est émise.
2. Le Raspberry Pi active au besoin les LED d'éclairage selon la luminosité.
3. La main est capturée et analysée.
4. Le résultat est comparé au profil stocké.
5. Le système pilote LED, buzzer et LCD selon la décision.
6. L'événement est publié en MQTT et stocké localement / dans Firebase.

### Administration mobile

1. Connexion par email / mot de passe.
2. Chargement des préférences propres à l'utilisateur.
3. Consultation des utilisateurs, accès et audits.
4. Mise à jour des paramètres du système en temps réel via MQTT.

## Répertoires

- `iot/` : backend Raspberry Pi
- `Mobile/` : application mobile Expo
- `docs/` : documentation projet
- `Maquettes/` : captures de maquettes visuelles du design mobile

## Choix techniques

- MQTT : commandes bidirectionnelles simples et démonstration IoT directe
- Firebase : auth et stockage distant sans backend HTTP séparé
- SQLite : continuité locale côté Pi en cas de coupure réseau
- OpenCV : traitement déterministe, léger et explicable

## Séparation backend / mobile

### Backend IoT

Le backend IoT exécute :

- la logique matérielle
- les lectures de capteurs
- l'analyse biométrique
- la persistance locale
- la publication MQTT

### Application mobile

L'application mobile exécute :

- la connexion utilisateur
- l'administration des profils
- l'affichage des données
- la configuration distante du système
- la persistance des préférences utilisateur
