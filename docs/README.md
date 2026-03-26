# Documentation BioGuard Access

Ce dossier centralise la documentation fonctionnelle, technique et académique du projet.

## Index

- [Documentation_Projet_BioGuard.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/Documentation_Projet_BioGuard.md) : dossier principal du projet avec séparation explicite entre backend IoT et application mobile
- [01_Architecture_Globale.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/01_Architecture_Globale.md) : architecture générale, frontières entre backend et mobile, composants et flux
- [02_Backend_IoT.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/02_Backend_IoT.md) : documentation détaillée du backend Raspberry Pi
- [03_Application_Mobile.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/03_Application_Mobile.md) : documentation détaillée de l'application mobile Expo
- [04_MQTT_Et_Donnees.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/04_MQTT_Et_Donnees.md) : contrat d'échange entre mobile et backend, topics MQTT, structures de données, stockage cloud et local
- [05_Installation_Execution.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/05_Installation_Execution.md) : installation et lancement, séparés pour backend IoT et application mobile
- [06_Demonstration_Et_Conformite.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/06_Demonstration_Et_Conformite.md) : scénario de démo, conformité, écarts backend et mobile

## Portée actuelle

Le projet repose sur :

- un Raspberry Pi avec caméra, capteur de lumière, buzzer, LCD et LED
- une application mobile Expo en JavaScript
- MQTT pour les commandes temps réel
- Firebase pour l'authentification et le stockage cloud
- SQLite comme cache local côté IoT

Les documents ci-dessous décrivent l'état réel du dépôt, pas une version théorique.
