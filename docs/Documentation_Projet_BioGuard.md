# Documentation Projet BioGuard Access

## 1. Résumé du projet

BioGuard Access est un système de contrôle d'accès intelligent conçu autour d'un Raspberry Pi. Le prototype identifie une personne avec une combinaison de biométrie palmaire et de géométrie des doigts, mesure la luminosité ambiante, puis transmet les événements vers une application mobile via MQTT. Les données de profils, d'historique, de télémétrie et de préférences utilisateur sont synchronisées vers Firebase.

Le projet répond aux exigences IoT et mobile en combinant :

- électronique embarquée
- application mobile
- communication réseau temps réel
- stockage cloud
- retour utilisateur local sur le prototype physique

## 2. Problème ciblé

Les petites entreprises, laboratoires, salles d'équipement et locaux sensibles ont souvent des solutions d'accès soit trop simples, soit trop coûteuses. Un code PIN peut être partagé. Une seule modalité biométrique peut être peu robuste. BioGuard Access propose un contrôle d'accès plus fiable, plus explicable et plus abordable.

## 3. Client cible

- PME
- laboratoires scolaires
- salles de serveurs
- bureaux administratifs
- résidences ou bâtiments avec zones restreintes

## 4. Proposition de valeur

- authentification plus robuste avec plusieurs signaux
- prototype faisable sur Raspberry Pi
- historique consultable sur mobile
- mode local même si Internet tombe
- architecture simple à démontrer devant jury

## 5. Capteurs et actionneurs

### Capteurs

1. Caméra Raspberry Pi : capture de la paume et des doigts
2. Capteur de lumière : mesure la luminosité pour piloter l'éclairage d'appoint

### Actionneurs

- LED verte : accès autorisé
- LED rouge : accès refusé
- deux LED d'éclairage : assistance visuelle si la pièce est sombre
- buzzer : retour sonore
- écran LCD I2C : instructions, état et messages

## 6. Architecture technique

```text
Application mobile Expo
        |
        | MQTT
        v
Passerelle Raspberry Pi
        |- contrôleur matériel
        |- service biométrique déterministe
        |- cache SQLite
        |- synchronisation Firebase
        v
Porte / système d'accès
```

## 6.1 Backend IoT

Le backend IoT correspond au Raspberry Pi et à tous les modules Python du dossier `iot/`.

Responsabilités :

- lecture des capteurs réellement disponibles
- pilotage du LCD, des LED et du buzzer
- capture biométrique via la caméra
- comparaison biométrique locale
- journalisation locale SQLite
- synchronisation Firebase
- exposition des commandes et réponses via MQTT

Modules principaux :

- `mqtt_gateway.py`
- `core/security_controller.py`
- `hardware/`
- `database.py`
- `cloud/firebase_service.py`

## 6.2 Application mobile

L'application mobile correspond au projet Expo du dossier `Mobile/`.

Responsabilités :

- authentification Firebase
- affichage des utilisateurs, événements et audits
- CRUD utilisateur
- consultation de la télémétrie
- envoi des commandes MQTT au backend IoT
- gestion des préférences propres à l'utilisateur

Modules principaux :

- `App.js`
- `navigation/NavigationRoot.js`
- `store/authStore.js`
- `store/mqttStore.js`
- `services/firebase.js`
- `ecrans/`

## 7. Algorithme biométrique

Le choix retenu est un algorithme déterministe, explicable et léger :

- segmentation de la main dans une ROI centrale
- extraction des moments de Hu
- extraction de mesures géométriques : aire, périmètre, solidité, ratio, convexité
- signature ORB moyenne pour la texture de la paume

Pourquoi ce choix :

- compatible Raspberry Pi
- plus simple à expliquer qu'un gros modèle IA
- plus rapide à ajuster pour un prototype de session
- permet un score de décision compréhensible

## 8. Flux d'utilisation

### Enrôlement

1. L'administrateur ajoute un utilisateur dans l'app mobile.
2. L'app envoie une commande MQTT d'enrôlement.
3. Le Raspberry Pi guide l'utilisateur avec le LCD.
4. Le Pi capture la paume et les doigts.
5. Un profil biométrique est créé.
6. Le profil est sauvegardé localement et synchronisé vers Firebase.

### Accès

1. L'utilisateur se présente devant la porte.
2. Le Pi vérifie la luminosité ambiante et active au besoin les LED d'éclairage.
3. Le LCD demande la paume et le positionnement des doigts.
4. Le Pi capture les données et calcule un score.
5. Si le score est sous le seuil, LED verte + buzzer court.
6. Sinon LED rouge + buzzer multiple.
7. L'événement est journalisé localement puis synchronisé vers Firebase.

## 9. IoT + mobile

### Côté IoT

- Python pour les capteurs et la logique
- MQTT pour recevoir les commandes
- SQLite pour le cache edge
- Firebase Admin SDK pour synchroniser

### Côté mobile

- Expo / React Native
- Zustand pour l'état local
- MQTT pour les actions temps réel
- Firebase pour les données cloud et l'administration
- `AsyncStorage` et `SecureStore` pour la session et les préférences locales
- `FlatList` avec icônes pour les utilisateurs, l'historique et les audits
- authentification email / mot de passe avec création de compte et session persistante optionnelle

## 9.1 Frontière backend / mobile

Le backend IoT n'expose pas une API HTTP. La frontière entre les deux couches repose sur :

- MQTT pour les commandes et retours temps réel
- Firebase pour l'authentification et le stockage cloud

En pratique :

- le backend IoT exécute la logique métier matérielle
- le mobile agit comme client d'administration et d'observation

## 10. Utilisation de Firebase

Firebase est le backend de stockage applicatif.

Collections proposées :

- `users`
- `biometric_profiles`
- `access_events`
- `device_telemetry`
- `mobile_user_preferences`

Rôle de Firebase :

- centraliser les profils utilisateurs
- stocker l'historique des accès
- conserver les événements administratifs
- permettre l'affichage mobile même à distance
- authentifier les administrateurs et opérateurs par email / mot de passe avec Firebase Authentication

Le Raspberry Pi garde aussi un cache SQLite pour éviter qu'une panne réseau bloque totalement le prototype.

## 11. Topics MQTT

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

## 12. Structure du code

### Dossier `iot`

- `config.py` : variables globales et GPIO
- `database.py` : tables locales SQLite
- `mqtt_gateway.py` : passerelle principale
- `core/security_controller.py` : orchestration des capteurs et actionneurs
- `hardware/` : LCD, LED, buzzer, capteurs et caméra
- `biometrics/biometrics_service.py` : traitement d'image et matching
- `cloud/firebase_service.py` : synchronisation Firebase

### Dossier `Mobile`

- `store/mqttStore.js` : client MQTT et requêtes
- `services/firebase.js` : initialisation Firebase
- `ecrans/` : login, dashboard, enrôlement, historique, scan

## 13. Démonstration devant jury

Séquence recommandée :

1. Montrer l'écran mobile, la connexion Firebase et la session persistante.
2. Présenter le prototype physique avec LCD, LEDs et buzzer.
3. Enrôler un utilisateur.
4. Modifier un utilisateur puis le supprimer.
4. Faire une tentative d'accès valide.
5. Faire une tentative d'accès refusée.
6. Montrer l'historique dans l'application.
7. Montrer les paramètres par utilisateur et le contrôle des actionneurs.
8. Expliquer que Firebase sert de backend et SQLite de cache local.

## 14. Coût de production estimé

| Composant | Coût approx. |
|---|---:|
| Raspberry Pi | 90 $ |
| Caméra Pi | 30 $ |
| Capteur de lumière | 3 $ |
| LCD I2C | 10 $ |
| LEDs + résistances + buzzer | 8 $ |
| Boîtier imprimé 3D | 20 $ |
| Alimentation et câblage | 15 $ |
| **Total** | **176 $** |

## 15. Prix de vente proposé

- Prix de vente : 399 $

Justification :

- laisse une marge intéressante
- reste nettement plus abordable qu'un système industriel
- compatible avec un achat PME / laboratoire / établissement scolaire

## 16. Marge de profit

- Coût unitaire estimé : 176 $
- Prix de vente : 399 $
- Profit unitaire : 223 $
- Profit sur 1000 unités : 223 000 $

## 17. Investissement demandé au Dragon

- Montant demandé : 50 000 $
- Pourcentage offert : 15 %

Utilisation prévue :

- industrialisation du boîtier 3D
- certification et sécurité logicielle
- achat de composants en volume
- amélioration de la précision de capture paume/doigts et de la partie mobile

## 18. Points forts du prototype

- aligné avec les contraintes IoT
- mobile + MQTT + Firebase
- prototype démontrable en direct
- logique biométrique explicable
- bonne séparation entre edge local et backend cloud
- exigences mobiles couvertes : auth, paramètres, listes, stockage local, CRUD

## 19. Limites actuelles

- la qualité de la segmentation paume/doigts dépend encore de l'éclairage et du positionnement
- Firebase exige des clés de configuration non incluses dans le dépôt
- les seuils biométriques doivent être calibrés avec vos propres essais sur Raspberry Pi

## 20. Conclusion

BioGuard Access est une version plus réaliste et plus forte commercialement de votre idée initiale. Le projet combine maintenant matériel, IoT, mobile, stockage cloud et logique biométrique multimodale dans une architecture adaptée à une présentation finale de type Dragons' Den.
