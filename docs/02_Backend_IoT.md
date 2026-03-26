# Backend IoT

## 1. Rôle du backend

Le backend IoT est la brique qui transforme le Raspberry Pi en passerelle de contrôle d'accès. C'est lui qui :

- reçoit les commandes MQTT
- interagit avec le matériel
- déclenche et orchestre les captures biométriques
- prend la décision d'accès
- journalise les événements
- synchronise vers Firebase quand c'est possible

Le backend se trouve dans le dossier `iot/`.

## 2. Vue d'ensemble du dossier `iot/`

| Fichier / dossier | Rôle |
|---|---|
| `config.py` | constantes applicatives, paramètres MQTT, Firebase, caméra, GPIO et biométrie |
| `mqtt_gateway.py` | point d'entrée principal, handlers MQTT, publication du statut et de la télémétrie |
| `core/security_controller.py` | coordination entre matériel, preview, capture et feedback local |
| `biometrics/biometrics_service.py` | analyse de la main, extraction PalmCode, fusion et matching |
| `database.py` | base SQLite locale, tables, audit et journalisation |
| `cloud/firebase_service.py` | couche facultative Firestore via `firebase-admin` |
| `hardware/` | abstractions capteurs et actionneurs |
| `requirements.txt` | dépendances Python du backend |

## 3. Point d'entrée principal

Le point d'entrée est `iot/mqtt_gateway.py`.

Ce module :

- initialise SQLite
- construit `SecurityController`
- initialise `FirebaseService`
- se connecte au broker MQTT
- souscrit au wildcard de commandes
- distribue les messages reçus vers les handlers dédiés
- publie périodiquement la télémétrie

Le backend tourne donc comme une boucle événementielle légère, fondée sur MQTT plutôt que sur un serveur HTTP.

## 4. Contrôleur matériel

Le module `iot/core/security_controller.py` orchestre le matériel.

### 4.1 Matériel géré

- LED verte
- LED rouge
- LED d'éclairage 1
- LED d'éclairage 2
- buzzer
- LCD I2C
- capteur de lumière
- caméra

### 4.2 Responsabilités du contrôleur

- séquence de boot visuelle et sonore
- état de repos du système
- synchronisation automatique de l'éclairage
- démarrage / arrêt des sessions de preview
- préparation des captures de scan ou d'enrôlement
- pilotage des retours utilisateur après succès ou refus
- construction d'une télémétrie structurée

### 4.3 États utiles

Le contrôleur maintient notamment :

- `auto_light_enabled`
- `manual_light_enabled`
- `preview_stream_enabled`

Ces états peuvent être modifiés à distance depuis l'application mobile.

## 5. Couche matérielle

Le dossier `iot/hardware/` fournit des abstractions propres pour le matériel.

### 5.1 Caméra

La caméra est gérée par `hardware/camera.py`.

Fonctions principales :

- capture d'une image BGR avec `capture_array()`
- génération d'une preview JPEG base64
- sauvegarde d'une capture sur disque
- snapshot d'état caméra
- fallback mock en l'absence de `Picamera2` ou en `VG_MOCK_MODE=1`

Le snapshot caméra contient par exemple :

- `available`
- `mock_mode`
- `width`
- `height`
- `frame_duration_us`
- `contrast`
- `sharpness`

### 5.2 Capteur de lumière

Le capteur de lumière est géré par `hardware/sensor.py`.

Il fournit :

- une lecture moyenne par méthode RC
- une calibration de baseline
- un test `is_dark()`
- un ratio de noirceur ajustable
- un snapshot structuré

Le snapshot retourne :

- `value`
- `baseline`
- `dark_ratio`
- `dark_threshold`
- `is_dark`

### 5.3 LCD

Le LCD I2C est géré par `hardware/lcd.py`.

Fonctions principales :

- initialisation sur `0x27` avec fallback `0x3F`
- affichage de deux lignes
- scrolling simple
- extinction / nettoyage propre
- snapshot contenant l'adresse, l'état et les deux lignes courantes

### 5.4 LED et buzzer

Le projet distingue :

- des LED pilotées individuellement
- un buzzer avec actions `on`, `off`, `beep`

Ces modules exposent également un `snapshot()` pour la télémétrie.

## 6. Pipeline biométrique

Le backend n'utilise pas de modèle de deep learning. Il repose sur une chaîne déterministe implémentée dans `iot/biometrics/biometrics_service.py`.

## 6.1 Étapes globales

1. prétraitement de l'image
2. segmentation de la main
3. extraction du contour principal
4. recherche des vallées entre les doigts
5. alignement anatomique et extraction de la ROI palmaire
6. amélioration locale du contraste des lignes
7. filtrage Gabor multi-orientation
8. découpage en anneaux concentriques
9. calcul des statistiques par anneau
10. création du vecteur PalmCode
11. estimation de la qualité de capture
12. comparaison avec les profils d'enrôlement

## 6.2 Caractéristiques du PalmCode utilisé

Le code actuel suit une logique de PalmCode plus classique que les versions antérieures du projet.

Le backend calcule :

- une ROI palmaire normalisée
- un ensemble de réponses de Gabor sur au moins 6 orientations
- des statistiques `mean + variance` par anneaux concentriques
- une similarité cosinus entre vecteurs

Le profil généré contient :

- `palmprint.geometry`
- `palmprint.intensity_histogram`
- `palmprint.palmcode_vector`
- `palmprint.palmcode_metadata`
- `palmprint.quality`
- `palmprint.alignment`

## 6.3 Qualité de capture

Le backend évalue la qualité avant d'accepter une capture exploitable.

Les dimensions prises en compte incluent :

- ratio de remplissage du masque
- force de réponse des lignes
- confiance orientationnelle des réponses de Gabor
- netteté
- cohérence géométrique de la main

Cette étape évite d'accepter des captures trop floues, trop excentrées ou mal segmentées.

## 6.4 Enrôlement et fusion

Lors d'un enrôlement :

- plusieurs images sont capturées
- les captures invalides sont rejetées
- les échantillons valides sont fusionnés
- le vecteur PalmCode final est obtenu par moyenne des vecteurs individuels

Le profil fusionné contient aussi :

- `samples`
- `sample_count`
- `captured_frame_count`
- `rejected_samples`
- `fusion_mode`

## 6.5 Matching

Le backend distingue deux cas :

- si les profils contiennent un `palmcode_vector`, le matching principal utilise la similarité PalmCode
- sinon, un fallback legacy exploite les descripteurs anciens de forme et de géométrie

Cela permet une compatibilité minimale avec d'anciens profils tout en privilégiant le pipeline moderne.

## 7. Commandes MQTT gérées par le backend

Le backend écoute notamment :

- `bioguard/cmd/access/scan`
- `bioguard/cmd/users/enroll`
- `bioguard/cmd/auth/login`
- `bioguard/cmd/users/list`
- `bioguard/cmd/users/update`
- `bioguard/cmd/users/delete`
- `bioguard/cmd/access/logs`
- `bioguard/cmd/audit/list`
- `bioguard/cmd/settings/update`
- `bioguard/cmd/ping`
- `bioguard/cmd/camera/preview`

Chaque handler :

- lit le `client_id`
- construit le topic de réponse dédié
- exécute la logique
- publie une réponse JSON

## 8. Persistance locale SQLite

Le backend initialise SQLite via `database.init_db()`.

### 8.1 Tables locales

- `users`
- `biometric_profiles`
- `access_events`
- `audit_logs`
- `device_state`

### 8.2 Contenu des tables

#### `users`

Contient :

- identifiant utilisateur
- nom d'utilisateur
- email
- mot de passe hashé
- rôle
- département
- éventuel `firebase_uid`
- date de création

#### `biometric_profiles`

Contient :

- `user_id`
- `profile_json`
- date de mise à jour

#### `access_events`

Contient :

- identifiant d'événement
- utilisateur
- statut
- score
- raison
- méthode
- modalités
- drapeau de synchronisation
- horodatage

#### `audit_logs`

Contient :

- niveau
- titre
- description
- métadonnées
- horodatage

#### `device_state`

Utilisée pour mémoriser des états tels que :

- dernières télémétries
- réglages appliqués
- valeurs de retour matériel

## 9. Synchronisation Firebase

Le backend utilise `firebase-admin` via `iot/cloud/firebase_service.py`.

Le service s'active seulement si :

- `VG_FIREBASE_ENABLED=1`
- les dépendances Firebase sont installées
- le fichier de credentials existe

Les opérations cloud disponibles incluent :

- sauvegarde / suppression de profils utilisateurs
- sauvegarde / suppression de profils biométriques
- lecture d'un profil biométrique
- sauvegarde d'événements d'accès
- sauvegarde de télémétrie

Si Firebase n'est pas disponible, le backend reste fonctionnel localement.

## 10. Cycle d'un scan d'accès

Le flux réel côté backend est le suivant.

1. réception d'une commande `scan`
2. passage du système en mode analyse
3. capture d'une image et d'une télémétrie
4. construction d'un profil live
5. recherche d'un profil cible
6. calcul de la décision
7. pilotage des actionneurs
8. journalisation dans SQLite
9. publication de l'événement et de la réponse

En cas d'échec, le backend produit aussi une raison explicite telle que :

- capture invalide
- profil absent
- mismatch biométrique

## 11. Cycle d'un enrôlement

Le flux réel côté backend est le suivant.

1. réception d'une commande `users/enroll`
2. démarrage du mode enrôlement
3. affichage des consignes sur LCD
4. capture de plusieurs frames
5. construction du profil fusionné
6. insertion / mise à jour du compte utilisateur
7. sauvegarde du profil biométrique
8. synchronisation Firebase
9. écriture d'un audit
10. réponse détaillée au mobile

## 12. Télémétrie backend

La télémétrie publiée par le backend contient notamment :

- `captured_at`
- `device_id`
- `light_sensor`
- `lighting`
- `buzzer`
- `lcd`
- `camera`

La preview caméra peut être ajoutée à cette télémétrie sous forme de JPEG base64 pendant les sessions de prévisualisation.

## 13. Gestion d'erreurs et mode dégradé

Le backend gère plusieurs cas d'erreur.

### 13.1 Erreurs d'entrée

- JSON invalide
- identifiants manquants
- `user_id` absent sur une opération qui l'exige

### 13.2 Erreurs biométriques

- absence de main
- capture floue
- segmentation invalide
- profil inexistant
- score insuffisant

### 13.3 Erreurs matérielles

- caméra absente
- GPIO indisponible
- LCD indisponible
- capteur non initialisé

### 13.4 Stratégies de survie

- mock mode si matériel indisponible
- SQLite si Firebase indisponible
- réponses d'erreur MQTT explicites pour le mobile

## 14. Paramètres importants dans `config.py`

Le fichier `iot/config.py` centralise les réglages principaux.

Catégories majeures :

- topics MQTT
- ports réseau
- identifiants MQTT
- dimensions caméra
- réglages NoIR / capture
- paramètres PalmCode
- seuils de validation biométrique
- GPIO
- options Firebase

Exemples de paramètres importants :

- `VG_MOCK_MODE`
- `VG_MQTT_BROKER`
- `VG_MQTT_PORT`
- `VG_MQTT_WS_PORT`
- `VG_FIREBASE_ENABLED`
- `VG_PALM_CODE_ORIENTATIONS`
- `VG_PALM_CODE_RING_COUNT`
- `VG_PALM_CODE_RING_OVERLAP`
- `VG_PALMCODE_MATCH_THRESHOLD`

## 15. Forces du backend

Les points forts du backend actuel sont :

- structure modulaire lisible
- séparation claire entre MQTT, matériel, biométrie et base
- mode local robuste pour un prototype
- pipeline biométrique explicable
- télémétrie détaillée utile pour la démo et le debug

## 16. Limites et dette technique

Les limites importantes à connaître sont :

- absence d'API HTTP d'administration
- sécurité applicative volontairement simple
- besoin de recalibrage fin des seuils biométriques sur le vrai matériel
- dépendance aux conditions d'éclairage et au cadrage de la main
- quelques noms historiques encore orientés `vein` dans le mobile alors que le backend est désormais centré paume

## 17. Conclusion

Le backend IoT de BioGuard Access constitue une base solide pour un prototype académique avancé :

- il est autonome
- il est modulaire
- il dialogue correctement avec le mobile
- il prend une décision locale
- il reste exploitable même en environnement partiellement dégradé

Sa compréhension est essentielle pour toute maintenance ou démonstration sérieuse du projet.
