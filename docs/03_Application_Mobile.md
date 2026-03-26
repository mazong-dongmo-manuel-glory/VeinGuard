# Application Mobile

## Stack

- Expo / React Native
- JavaScript
- React Navigation
- Zustand
- Firebase Authentication
- Firestore
- MQTT
- AsyncStorage
- SecureStore

## Rôle de l'application

Le mobile sert de console d'administration et de supervision. Il permet de :

- se connecter avec Firebase
- créer un compte
- conserver la session selon le choix utilisateur
- configurer l'adresse du Raspberry Pi et les ports MQTT / WebSocket
- gérer les utilisateurs
- consulter l'historique d'accès
- consulter les logs d'audit
- piloter certaines fonctions du Raspberry Pi
- stocker les préférences utilisateur

## Écrans

- `Login` : connexion, création de compte, reset de mot de passe
- `Dashboard` : état global du système
- `AccessHistory` : historique d'accès
- `AccessEvent` : détail d'un événement
- `AdminAuditLogs` : journal d'audit
- `UserManagement` : liste des utilisateurs
- `EnrollUser` : ajout / modification d'un utilisateur
- `VeinScanBiometrics` : lancement de scan et interface biométrique
- `SystemSetting` : commande matérielle et préférences applicatives

## Authentification

L'application utilise Firebase Authentication avec :

- email / mot de passe
- création de compte
- réinitialisation de mot de passe
- garde de navigation si l'utilisateur n'est pas connecté
- option "rester connecté"

Les identifiants de session persistante sont gérés avec `SecureStore` et le drapeau de persistance avec `AsyncStorage`.

## Configuration du serveur Raspberry Pi

L'application mobile permet de modifier la connexion MQTT :

- dès l'écran de connexion
- depuis l'écran des paramètres

Les éléments configurables sont :

- l'adresse IP ou le nom d'hôte du Raspberry Pi
- le port WebSocket MQTT utilisé par le mobile
- le port MQTT TCP de référence du backend

Cette configuration est persistée localement et la reconnexion MQTT est relancée automatiquement après modification.

## Préférences utilisateur

Chaque utilisateur dispose de préférences rechargées à la connexion :

- actualisation automatique des données
- affichage compact des listes
- affichage des détails techniques

Ces préférences sont :

- mises en cache localement
- associées à l'utilisateur connecté
- synchronisées vers Firestore

## CRUD utilisateur

Le mobile permet :

- ajout d'utilisateur
- modification d'utilisateur
- suppression d'utilisateur
- affichage de la liste synchronisée depuis le backend IoT

## Listes

Les listes principales utilisent `FlatList` avec des icônes :

- utilisateurs
- historique d'accès
- audits

## Contraintes UX traitées

- interface en français
- prévention des débordements par `flexShrink`, `numberOfLines` et cartes compactes
- messages d'erreur plus explicites pour Firebase
- rafraîchissement manuel et automatique des listes
