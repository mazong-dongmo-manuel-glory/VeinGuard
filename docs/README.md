# Documentation BioGuard Access

Ce dossier regroupe la documentation fonctionnelle, technique et de soutenance du projet `VeinGuard`, dont l'application et le prototype exposent la marque produit `BioGuard Access`.

La documentation a été réécrite pour refléter l'état réel du dépôt au `26 mars 2026`, avec une séparation claire entre :

- le backend IoT exécuté sur Raspberry Pi
- l'application mobile Expo / React Native
- les échanges MQTT
- les données locales et cloud
- les procédures d'installation, de démonstration et d'évaluation

## Objectif du dossier

Ce dossier sert à quatre usages complémentaires :

1. comprendre rapidement le projet dans son ensemble
2. installer et exécuter le système sur un environnement de démonstration
3. maintenir le code côté Raspberry Pi et côté mobile
4. préparer une présentation académique ou professionnelle du prototype

## Lecture recommandée

Selon le besoin, l'ordre de lecture conseillé est le suivant.

### Pour découvrir le projet

1. [Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md)
2. [01_Architecture_Globale.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/01_Architecture_Globale.md)
3. [06_Demonstration_Et_Conformite.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/06_Demonstration_Et_Conformite.md)

### Pour travailler sur le backend Raspberry Pi

1. [01_Architecture_Globale.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/01_Architecture_Globale.md)
2. [02_Backend_IoT.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/02_Backend_IoT.md)
3. [04_MQTT_Et_Donnees.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/04_MQTT_Et_Donnees.md)
4. [05_Installation_Execution.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/05_Installation_Execution.md)

### Pour travailler sur l'application mobile

1. [01_Architecture_Globale.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/01_Architecture_Globale.md)
2. [03_Application_Mobile.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/03_Application_Mobile.md)
3. [04_MQTT_Et_Donnees.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/04_MQTT_Et_Donnees.md)
4. [05_Installation_Execution.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/05_Installation_Execution.md)

### Pour préparer une démo, un rapport ou une soutenance

1. [Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md)
2. [06_Demonstration_Et_Conformite.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/06_Demonstration_Et_Conformite.md)
3. [05_Installation_Execution.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/05_Installation_Execution.md)

## Carte des documents

| Fichier | Rôle |
|---|---|
| [Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md) | dossier maître du projet, utile pour la vision d'ensemble, le contexte et la présentation |
| [01_Architecture_Globale.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/01_Architecture_Globale.md) | vue système, séparation backend/mobile, composants, flux fonctionnels et contraintes |
| [02_Backend_IoT.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/02_Backend_IoT.md) | description détaillée du backend Python sur Raspberry Pi, du matériel et de la biométrie |
| [03_Application_Mobile.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/03_Application_Mobile.md) | documentation détaillée de l'application mobile Expo, de ses écrans et de ses stores |
| [04_MQTT_Et_Donnees.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/04_MQTT_Et_Donnees.md) | contrat d'échange temps réel, payloads MQTT, structures de données, stockage SQLite et Firestore |
| [05_Installation_Execution.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/05_Installation_Execution.md) | guide d'installation, de configuration et d'exécution pour le Pi, le broker et le mobile |
| [06_Demonstration_Et_Conformite.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/06_Demonstration_Et_Conformite.md) | script de démonstration, points forts, écarts et lecture de conformité par rapport au sujet |

## Résumé rapide du projet

BioGuard Access est un prototype de contrôle d'accès intelligent composé de deux sous-systèmes.

- Un backend IoT, dans le dossier `iot/`, pilote le matériel du Raspberry Pi, capture la paume, calcule un profil biométrique local et expose ses actions via MQTT.
- Une application mobile, dans le dossier `Mobile/`, permet à un opérateur ou à un administrateur de se connecter, gérer les utilisateurs, lancer un enrôlement, déclencher un scan et superviser le système.

Le projet utilise :

- `MQTT` pour les commandes et retours en temps réel
- `Firebase Authentication` pour la connexion mobile
- `Firestore` pour la synchronisation cloud
- `SQLite` pour la continuité locale côté Raspberry Pi
- `OpenCV` pour le traitement biométrique déterministe

## État fonctionnel actuel

Le dépôt intègre aujourd'hui :

- un backend Python structuré autour de `mqtt_gateway.py`
- une couche matérielle dédiée aux LED, au buzzer, au LCD, au capteur de lumière et à la caméra
- un pipeline biométrique basé sur une ROI de paume alignée puis un vecteur de type `PalmCode`
- une application mobile Expo avec navigation par onglets, authentification Firebase, configuration MQTT et écrans de supervision
- une synchronisation cloud simple mais exploitable pour les profils utilisateurs, les profils biométriques, les événements d'accès et la télémétrie

## Point important sur la biométrie

Le projet a évolué d'une logique plus hybride vers un pipeline davantage centré sur la paume.

Le code courant documenté ici repose sur :

- la segmentation de la main
- l'extraction d'une ROI palmaire normalisée
- un filtrage multi-orientation par noyaux de Gabor
- une extraction de statistiques `moyenne + variance` sur anneaux concentriques
- une comparaison par similarité cosinus entre vecteurs PalmCode

Des caractéristiques auxiliaires de forme de la main restent présentes dans le profil pour le debug, la visualisation et la compatibilité avec des profils plus anciens, mais le cœur du matching moderne s'appuie désormais sur le `PalmCode`.

## Référentiel du dépôt

Les répertoires les plus importants sont :

- `iot/` : backend Raspberry Pi, matériel, biométrie, base locale et Firebase
- `Mobile/` : application Expo / React Native
- `docs/` : documentation complète
- `Maquettes/` : éléments visuels du design ou du support de présentation

## Glossaire rapide

- `ROI` : region of interest, zone de l'image réellement utilisée pour l'analyse
- `PalmCode` : représentation vectorielle de la texture de paume extraite après filtrage multi-orientation
- `MQTT` : protocole léger de messagerie publish/subscribe, adapté aux systèmes IoT
- `Firestore` : base NoSQL cloud utilisée ici comme stockage applicatif
- `SQLite` : base embarquée locale côté Pi
- `Preview` : prévisualisation caméra envoyée au mobile via la télémétrie
- `Mock mode` : mode d'exécution sans matériel physique complet, utile pour le développement

## Principe de vérité documentaire

Cette documentation décrit :

- l'architecture réellement présente dans le dépôt
- les noms de fichiers réellement utilisés
- les topics MQTT réellement déclarés dans la configuration
- les flux réellement codés dans l'application mobile et le backend

Quand un point du sujet académique n'est pas totalement couvert par le prototype, cela est indiqué explicitement au lieu d'être maquillé.
