# Application Mobile

## 1. Rôle de l'application

L'application mobile sert d'interface opérateur et d'outil de supervision du système BioGuard Access. Elle ne réalise pas le matching biométrique elle-même. Son rôle est de :

- authentifier l'utilisateur humain
- configurer la connexion au Raspberry Pi
- lancer des commandes via MQTT
- afficher les réponses et la télémétrie
- administrer les utilisateurs
- consulter l'historique et les audits
- exposer une interface moderne et démontrable

## 2. Stack technique

Le dossier `Mobile/` repose sur :

- Expo
- React Native
- JavaScript
- React Navigation
- Zustand
- Firebase Authentication
- Firestore
- `mqtt` pour le client MQTT
- `AsyncStorage`
- `expo-secure-store`
- `i18next`
- `expo-linear-gradient`
- `expo-blur`

## 3. Point d'entrée et bootstrap

Le point d'entrée applicatif est `Mobile/App.js`.

Au démarrage, l'application :

1. initialise l'internationalisation
2. charge la configuration MQTT sauvegardée
3. tente une connexion MQTT
4. initialise l'authentification Firebase
5. charge les préférences de l'utilisateur si une session existe
6. affiche un écran de chargement puis la navigation principale

L'application encapsule aussi son interface dans une `ErrorBoundary`, ce qui améliore la robustesse en démonstration.

## 4. Organisation du projet mobile

| Emplacement | Rôle |
|---|---|
| `App.js` | bootstrap général |
| `navigation/NavigationRoot.js` | structure de navigation principale |
| `store/authStore.js` | session, bootstrap auth, préférences utilisateur |
| `store/mqttStore.js` | connexion MQTT, requêtes, réponses, télémétrie |
| `services/firebase.js` | initialisation de Firebase |
| `services/cloudSync.js` | synchronisation Firestore côté mobile |
| `services/preferences.js` | chargement / sauvegarde des préférences |
| `services/auth.js` | messages d'erreur auth et reset mot de passe |
| `ecrans/` | écrans de l'application |
| `langues/` | traduction FR / EN |
| `config.js` | topics, hôte MQTT par défaut, configuration Firebase |

## 5. Navigation

La navigation est définie dans `Mobile/navigation/NavigationRoot.js`.

### 5.1 Logique générale

- si aucun utilisateur Firebase n'est connecté, l'application affiche `Login`
- si un utilisateur est connecté, l'application affiche une navigation principale par onglets

### 5.2 Onglets principaux

La navigation principale contient :

- `Dashboard`
- `AccessHistory`
- `VeinScan`
- `UserManagement`
- `SystemSetting`

Le bouton de scan central est traité comme un bouton visuellement accentué de type FAB.

### 5.3 Stacks secondaires

- un stack dédié à l'historique et au détail des événements
- un stack dédié à la gestion des utilisateurs et à l'écran d'enrôlement
- un écran modal de décision d'accès

## 6. Authentification

L'authentification est gérée par Firebase côté mobile, pas par le broker MQTT.

Le store `authStore.js` prend en charge :

- le bootstrap de session
- la connexion email / mot de passe
- la création de compte
- la déconnexion
- la persistance optionnelle des identifiants
- le chargement des préférences utilisateur

### 6.1 Persistance de session

Le comportement "rester connecté" est géré avec :

- `AsyncStorage` pour le drapeau de persistance
- `SecureStore` pour stocker email et mot de passe localement

### 6.2 Préférences utilisateur

Les préférences applicatives actuellement prévues sont :

- `autoRefreshData`
- `showTechnicalDetails`
- `compactLists`

Elles sont :

- chargées au login
- stockées en cache local
- synchronisées dans la collection `mobile_user_preferences` de Firestore

## 7. Connexion MQTT

Le store `mqttStore.js` centralise toute la logique temps réel.

### 7.1 Responsabilités

- charger la configuration réseau sauvegardée
- établir la connexion WebSocket au broker
- souscrire aux topics de statut, télémétrie et réponses
- publier les commandes
- corréler une réponse au bon `client_id`
- mettre à jour l'état global de l'application

### 7.2 État maintenu

Le store MQTT conserve notamment :

- `client`
- `isConnected`
- `gatewayOnline`
- `status`
- `statusPayload`
- `telemetry`
- `settingsAck`
- `lastScanResult`
- `lastError`
- `brokerConfig`
- `clientId`

### 7.3 Configuration réseau modifiable

L'application permet de modifier :

- l'hôte du Raspberry Pi ou du broker
- le port MQTT WebSocket
- le port MQTT TCP de référence
- le nom d'utilisateur MQTT
- le mot de passe MQTT

Cette configuration est persistée localement puis réappliquée à la reconnexion.

## 8. Synchronisation cloud côté mobile

En plus de la communication MQTT, le mobile utilise Firestore comme cache fonctionnel et comme source distante secondaire.

Le service `cloudSync.js` permet :

- de synchroniser les profils utilisateurs
- de synchroniser les profils biométriques
- de synchroniser les événements d'accès
- de synchroniser la télémétrie
- de supprimer des profils côté Firestore

Le store MQTT s'appuie dessus pour :

- conserver des données utiles lorsque MQTT échoue
- alimenter l'application en mode consultation

## 9. Écrans principaux

## 9.1 Login

L'écran `Login.js` regroupe :

- connexion
- création de compte
- réinitialisation de mot de passe
- option "rester connecté"
- configuration MQTT de base
- indicateur d'état du backend

Il combine donc deux préoccupations :

- l'accès applicatif via Firebase
- la connectivité temps réel vers le Raspberry Pi

## 9.2 Dashboard

L'écran `Dashboard.js` sert de synthèse système.

Il affiche notamment :

- le statut du gateway
- l'identifiant du dispositif
- l'état de la caméra
- l'état du capteur de lumière
- la dernière mise à jour reçue

Il fournit aussi des raccourcis vers :

- le scan
- l'historique
- la gestion des utilisateurs

## 9.3 VeinScanBiometrics

Malgré son nom historique, cet écran est aujourd'hui utilisé comme interface de scan pour le backend biométrique centré paume.

Fonctions principales :

- démarrage et arrêt de preview caméra
- lancement d'un scan d'identification
- saisie optionnelle d'un `user_id`
- affichage de la dernière image preview ou image traitée
- affichage d'indicateurs de qualité dérivés de la réponse

Cet écran est important pour la démonstration car il expose visuellement :

- le flux live
- le statut du système
- la présence du topic MQTT concerné

## 9.4 UserManagement

Cet écran permet :

- de charger la liste des utilisateurs
- de rechercher par nom, rôle ou email
- de rafraîchir manuellement
- d'éditer un utilisateur existant
- de supprimer un utilisateur
- d'accéder à l'écran d'enrôlement

L'affichage dépend aussi des préférences :

- mode compact ou non
- affichage des détails techniques
- auto-refresh

## 9.5 EnrollUser

Cet écran regroupe :

- les données de profil utilisateur
- la sélection d'un rôle
- le choix d'un département
- des groupes d'accès
- une zone de notes
- la prévisualisation caméra
- le déclenchement d'un enrôlement ou d'une mise à jour

En mode création :

- l'application envoie une commande complète d'enrôlement

En mode édition :

- l'application envoie une commande de mise à jour du profil

## 9.6 AccessHistory et AccessEvent

Ces écrans permettent de :

- consulter les événements d'accès
- afficher leur détail
- relier décision, score, statut et raisons

Ils sont particulièrement utiles pour démontrer :

- la traçabilité
- la continuité des données entre backend et mobile

## 9.7 AdminAuditLogs

Cet écran expose les journaux d'audit tels que :

- enrôlement utilisateur
- mise à jour d'un utilisateur
- suppression
- synchronisation de réglages

## 9.8 SystemSetting

L'écran des paramètres centralise trois familles de réglages.

### Réglages matériels

- éclairage automatique
- éclairage d'assistance
- LED rouge / verte
- seuil d'obscurité
- envoi de texte au LCD
- buzzer

### Préférences mobiles

- auto-refresh
- listes compactes
- détails techniques

### Paramètres réseau et langue

- hôte MQTT
- ports WebSocket / TCP
- changement de langue FR / EN

## 10. Internationalisation

Le projet utilise `i18next` avec deux langues :

- français
- anglais

L'état de langue est géré par `langueStore.js`, et les ressources se trouvent dans :

- `Mobile/langues/fr.json`
- `Mobile/langues/en.json`

## 11. Design et expérience utilisateur

L'application adopte une direction visuelle marquée :

- interface sombre
- effets de blur
- gradients
- icônes néon
- accent fort sur le scan et l'état du système

L'objectif est double :

- donner une identité visuelle forte à la démo
- rendre la supervision technique plus lisible

## 12. Flux mobile majeurs

### 12.1 Connexion

1. l'utilisateur saisit email et mot de passe
2. Firebase valide la session
3. les préférences sont rechargées
4. la navigation principale est déverrouillée

### 12.2 Enrôlement

1. l'utilisateur ouvre `EnrollUser`
2. remplit le formulaire
3. vérifie la preview caméra
4. envoie l'enrôlement
5. attend la réponse longue du backend
6. reçoit l'ID, la télémétrie et le profil associé

### 12.3 Scan

1. la preview caméra démarre
2. l'opérateur lance le scan
3. le mobile attend la réponse MQTT
4. en cas de succès, il navigue vers `AccessDecision`
5. en cas d'échec, il affiche une alerte détaillée

## 13. Forces de l'application mobile

- architecture simple à suivre
- bonne séparation entre stores, services et écrans
- configuration réseau modifiable sans recompilation
- intégration réelle de Firebase et MQTT
- écrans adaptés à une démonstration de projet

## 14. Limites connues

- quelques noms d'écrans historiques utilisent encore `Vein`
- la dépendance au broker MQTT est forte pour le temps réel
- la logique de cache et de synchronisation cloud reste volontairement simple
- l'application n'est pas pensée comme un client grand public distribué sur stores

## 15. Conclusion

L'application mobile joue bien son rôle de console de supervision et d'administration :

- elle n'empiète pas sur la logique biométrique
- elle expose les bonnes commandes
- elle rend le prototype démontrable
- elle permet d'observer la valeur du backend IoT

Pour comprendre les payloads et topics qu'elle utilise, il faut lire en complément [04_MQTT_Et_Donnees.md](/Users/mazong/Documents/GitHub/VeinGuard/docs/04_MQTT_Et_Donnees.md).
