# Architecture Globale

## 1. Objectif architectural

L'architecture de BioGuard Access a été conçue pour séparer proprement :

- la logique temps réel et matérielle
- l'interface d'administration mobile
- la communication réseau
- la persistance locale et cloud

Cette séparation permet :

- d'exécuter le contrôle d'accès sur le Raspberry Pi sans dépendre du mobile
- d'utiliser le mobile comme console de supervision et de commande
- de découpler l'interface utilisateur du traitement biométrique
- de garder une architecture simple à expliquer et à maintenir

## 2. Vue système

```text
                      +-----------------------------+
                      | Application mobile Expo     |
                      |-----------------------------|
                      | Firebase Auth               |
                      | Stores Zustand              |
                      | Écrans d'administration     |
                      | Client MQTT over WebSocket  |
                      +-------------+---------------+
                                    |
                                    | MQTT
                                    v
                      +-----------------------------+
                      | Broker MQTT                 |
                      +-------------+---------------+
                                    |
                                    | MQTT TCP
                                    v
                      +-----------------------------+
                      | Raspberry Pi / Backend IoT  |
                      |-----------------------------|
                      | mqtt_gateway.py             |
                      | SecurityController          |
                      | biometrics_service.py       |
                      | database.py                 |
                      | firebase_service.py         |
                      +------+------+------+--------+
                             |      |      |
                             |      |      +-------------------+
                             |      |                          |
                             v      v                          v
                     Matériel local   SQLite local       Firebase / Firestore
```

## 3. Principes de conception

### 3.1 Exécution locale prioritaire

La décision biométrique est prise côté Pi, pas dans l'application mobile. Ce point est important :

- la capture se fait localement
- le calcul biométrique se fait localement
- les LED, le LCD et le buzzer sont pilotés localement
- la panne du cloud ne doit pas empêcher la démo locale

### 3.2 Mobile comme console de commande

Le mobile n'est pas le cœur du contrôle d'accès. Il agit comme :

- interface opérateur
- console d'administration
- client de consultation
- point d'entrée pour les commandes MQTT

### 3.3 Cloud comme couche de synchronisation

Firebase n'est pas le moteur temps réel du système. Son rôle est :

- authentifier les utilisateurs mobiles
- conserver des données pour consultation distante
- synchroniser des profils et événements

### 3.4 Simplicité démontrable

L'architecture évite les couches inutiles.

Il n'y a pas de :

- backend HTTP séparé
- microservices multiples
- bus applicatif complexe
- modèle IA externe ou inférence distante

## 4. Blocs principaux

## 4.1 Backend IoT

Le backend IoT est la brique centrale du système.

Composants majeurs :

- `mqtt_gateway.py` : point d'entrée, boucle MQTT, handlers de commandes
- `core/security_controller.py` : orchestration des capteurs, actionneurs et sessions de capture
- `biometrics/biometrics_service.py` : traitement d'image, PalmCode, fusion, matching
- `database.py` : persistance SQLite
- `cloud/firebase_service.py` : persistance cloud facultative
- `hardware/*` : abstraction du matériel

## 4.2 Application mobile

L'application mobile est structurée autour de :

- `App.js` pour le bootstrap général
- `navigation/NavigationRoot.js` pour la navigation
- `authStore.js` pour l'authentification et les préférences
- `mqttStore.js` pour la connexion MQTT et les requêtes
- `services/*` pour Firebase, synchronisation et erreurs
- `ecrans/*` pour les vues utilisateur

## 4.3 Broker MQTT

Le broker MQTT joue le rôle de point de rendez-vous pour les échanges temps réel.

Il permet :

- l'envoi de commandes depuis le mobile
- la publication du statut du Pi
- la publication de la télémétrie
- la réception des réponses sur des topics corrélés au `client_id`

## 4.4 Persistances

Le système utilise deux niveaux de persistance.

### SQLite côté Pi

Utilisé pour :

- `users`
- `biometric_profiles`
- `access_events`
- `audit_logs`
- `device_state`

### Firestore

Utilisé pour :

- `users`
- `biometric_profiles`
- `access_events`
- `device_telemetry`
- `mobile_user_preferences`

## 5. Architecture physique

```text
Téléphone / tablette
    |
    | Wi-Fi / réseau local
    v
Broker MQTT
    |
    v
Raspberry Pi
    |- Caméra Pi
    |- Capteur de lumière
    |- LCD I2C
    |- Buzzer
    |- LED rouge
    |- LED verte
    |- LED éclairage 1
    |- LED éclairage 2
```

Le mobile et le Pi ne communiquent pas directement par USB ni par HTTP. Le réseau passe essentiellement par MQTT, avec Firebase comme couche cloud parallèle.

## 6. Flux d'initialisation

### 6.1 Démarrage du backend IoT

1. Initialisation de SQLite
2. Initialisation du contrôleur matériel
3. Séquence de boot visuelle et sonore
4. Initialisation du service Firebase si disponible
5. Connexion au broker MQTT
6. Souscription aux topics de commandes
7. Publication du statut `ONLINE`
8. Publication périodique de la télémétrie

### 6.2 Démarrage du mobile

1. Initialisation `i18next`
2. Chargement de la configuration MQTT locale
3. Tentative de connexion au broker en WebSocket
4. Bootstrap de l'authentification Firebase
5. Chargement des préférences utilisateur
6. Affichage de l'écran de connexion ou de l'application principale

## 7. Flux métier détaillés

## 7.1 Prévisualisation caméra

Le mobile peut ouvrir un preview avant un scan ou un enrôlement.

Séquence :

1. le mobile publie `bioguard/cmd/camera/preview`
2. le backend active la session de preview
3. le backend publie des trames JPEG base64 via la télémétrie
4. le mobile affiche l'image retournée
5. le mobile peut envoyer `action: stop` pour fermer la preview

## 7.2 Enrôlement biométrique

Séquence logique :

1. saisie d'un utilisateur dans l'écran mobile `EnrollUser`
2. envoi du payload d'enrôlement via MQTT
3. côté Pi, démarrage du mode `ENROLLMENT`
4. captures successives selon `ENROLLMENT_SAMPLE_COUNT`
5. construction d'un profil fusionné
6. sauvegarde locale SQLite
7. synchronisation optionnelle vers Firestore
8. retour d'une réponse MQTT contenant le profil, la télémétrie et les clés utiles

## 7.3 Identification

Séquence logique :

1. le mobile lance un scan via l'écran `VeinScanBiometrics`
2. le Pi force l'éclairage si nécessaire
3. une capture est analysée en profil live
4. le profil live est comparé aux profils enregistrés
5. le Pi pilote buzzer, LED et LCD
6. un événement d'accès est journalisé
7. la réponse de scan est renvoyée au mobile

## 7.4 Mise à jour de réglages

Le mobile peut envoyer une commande de réglages pour :

- activer ou désactiver l'éclairage automatique
- forcer les LED d'assistance
- allumer ou éteindre les LED rouge / verte
- déclencher un test buzzer
- écrire sur le LCD
- modifier le ratio d'obscurité

## 8. Architecture biométrique dans l'ensemble système

Le pipeline biométrique n'est pas une brique isolée. Il s'insère dans une chaîne complète.

```text
Caméra
  -> frame BGR
  -> prétraitement
  -> segmentation de la main
  -> extraction ROI de paume
  -> amélioration des lignes
  -> Gabor multi-orientation
  -> anneaux concentriques
  -> vecteur PalmCode
  -> comparaison
  -> score + décision
  -> actionneurs + événement + réponse MQTT
```

Le résultat n'est donc pas seulement un score. Il est exploité par :

- la logique d'accès
- la journalisation
- la télémétrie
- les écrans mobiles

## 9. Séparation des responsabilités

### 9.1 Ce qui appartient au backend

- toute interaction matérielle
- toute logique biométrique
- toute décision d'accès
- la vérité locale sur l'état du dispositif
- la continuité de service hors cloud

### 9.2 Ce qui appartient au mobile

- l'authentification opérateur
- la saisie utilisateur
- l'affichage et l'administration
- la configuration réseau MQTT
- la consultation de l'historique et des audits

### 9.3 Ce qui appartient au broker

- le transport des commandes et des réponses
- la diffusion de statut et de télémétrie

### 9.4 Ce qui appartient au cloud

- la synchronisation des données de référence
- la persistance distante
- la session applicative mobile via Firebase Authentication

## 10. Résilience et défaillances prises en compte

L'architecture prend en charge plusieurs situations dégradées.

### 10.1 Firebase indisponible

Le Pi peut continuer à fonctionner avec SQLite local uniquement.

### 10.2 Matériel partiellement absent

Le dépôt prévoit un `mock mode` pour permettre le développement sans matériel complet.

### 10.3 Mobile déconnecté

Le backend peut continuer à tourner, mais l'interface opérateur n'est plus disponible.

### 10.4 Broker indisponible

Le mobile perd le contrôle temps réel, mais le backend local peut toujours effectuer certaines actions si la démo est préparée autrement.

## 11. Points d'attention architecturaux

### 11.1 Nommage hérité

Certaines parties du mobile utilisent encore le mot `VeinScan` dans les noms d'écrans ou d'identifiants UI. Cela ne signifie pas que l'algorithme courant est de la reconnaissance veineuse. Le pipeline actuel documenté dans le backend est centré sur la paume.

### 11.2 Double persistance

Le projet écrit à la fois dans SQLite et, si possible, dans Firestore. Il faut donc documenter explicitement :

- quelle source est prioritaire pour une décision locale
- quelles données peuvent diverger temporairement
- quelles données sont simplement recopiées côté mobile

### 11.3 Seuils biométriques

L'architecture prévoit des seuils configurables. Ils sont adaptés au prototype et doivent être recalibrés si :

- le matériel change
- la lumière ambiante change fortement
- la distance caméra/main varie

## 12. Conclusion architecturale

L'architecture globale de BioGuard Access est volontairement simple, modulaire et orientée démonstration réelle.

Elle offre :

- une séparation nette des responsabilités
- un backend IoT autonome
- un mobile de supervision moderne
- une couche de communication temps réel claire
- une persistance hybride locale + cloud

Cette structure donne au projet une vraie cohérence système, ce qui est l'un de ses principaux points forts.
