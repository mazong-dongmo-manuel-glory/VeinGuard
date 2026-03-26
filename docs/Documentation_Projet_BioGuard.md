# Documentation Projet BioGuard Access

## 1. Résumé du projet

BioGuard Access est un système de contrôle d'accès intelligent conçu autour d'un Raspberry Pi. Le prototype identifie une personne avec une combinaison de biométrie palmaire et de géométrie des doigts, puis transmet les événements vers une application mobile via MQTT. Les données de profils, d'historique et de télémétrie sont synchronisées vers Firebase.

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
2. Capteur tactile / contact : confirme que l'utilisateur touche le point d'entrée
3. Capteur ultrasonique : détecte la présence devant le lecteur
4. Capteur PIR : détecte le mouvement près de la porte

### Actionneurs

- LED verte : accès autorisé
- LED rouge : accès refusé
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
2. Le capteur de présence et le capteur de mouvement s'activent.
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

## 10. Utilisation de Firebase

Firebase est le backend de stockage applicatif.

Collections proposées :

- `users`
- `biometric_profiles`
- `access_events`
- `device_telemetry`

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

1. Montrer l'écran mobile et l'état en ligne du système.
2. Présenter le prototype physique avec LCD, LEDs et buzzer.
3. Enrôler un utilisateur.
4. Faire une tentative d'accès valide.
5. Faire une tentative d'accès refusée.
6. Montrer l'historique dans l'application.
7. Expliquer que Firebase sert de backend et SQLite de cache local.

## 14. Coût de production estimé

| Composant | Coût approx. |
|---|---:|
| Raspberry Pi | 90 $ |
| Caméra Pi | 30 $ |
| LCD I2C | 10 $ |
| LEDs + résistances + buzzer | 8 $ |
| Capteur ultrasonique | 5 $ |
| Capteur PIR | 6 $ |
| Capteur tactile | 5 $ |
| Boîtier imprimé 3D | 20 $ |
| Alimentation et câblage | 15 $ |
| **Total** | **189 $** |

## 15. Prix de vente proposé

- Prix de vente : 399 $

Justification :

- laisse une marge intéressante
- reste nettement plus abordable qu'un système industriel
- compatible avec un achat PME / laboratoire / établissement scolaire

## 16. Marge de profit

- Coût unitaire estimé : 189 $
- Prix de vente : 399 $
- Profit unitaire : 210 $
- Profit sur 1000 unités : 210 000 $

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
- mobile + MQTT + cloud
- prototype démontrable en direct
- logique biométrique explicable
- bonne séparation entre edge local et backend cloud

## 19. Limites actuelles

- la qualité de la segmentation paume/doigts dépend encore de l'éclairage et du positionnement
- Firebase exige des clés de configuration non incluses dans le dépôt
- les seuils biométriques doivent être calibrés avec vos propres essais sur Raspberry Pi

## 20. Conclusion

BioGuard Access est une version plus réaliste et plus forte commercialement de votre idée initiale. Le projet combine maintenant matériel, IoT, mobile, stockage cloud et logique biométrique multimodale dans une architecture adaptée à une présentation finale de type Dragons' Den.
