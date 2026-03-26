# MQTT Et Données

## 1. Objectif de ce document

Ce document décrit :

- les topics MQTT réellement utilisés
- la convention de corrélation des réponses
- les payloads principaux envoyés par le mobile
- les réponses principales renvoyées par le backend
- les structures de données persistées en SQLite et Firestore

L'objectif est de disposer d'un contrat d'échange clair entre :

- l'application mobile
- le broker MQTT
- le backend Raspberry Pi
- les stockages locaux et cloud

## 2. Convention générale MQTT

Le projet utilise un préfixe de topic défini dans la configuration mobile et backend :

```text
bioguard
```

Les topics sont regroupés en trois familles.

### 2.1 Commandes

Elles sont publiées par le mobile vers le backend.

Format général :

```text
bioguard/cmd/<action>
```

### 2.2 Publications globales

Elles sont publiées par le backend pour diffuser son état.

Format général :

```text
bioguard/<canal>
```

### 2.3 Réponses corrélées

Le backend répond sur un topic construit à partir de la commande et du `client_id`.

Format général :

```text
bioguard/res/<commande>/<client_id>
```

Exemple :

```text
bioguard/res/users/enroll/mobile-a1b2c3d4
```

## 3. Topics utilisés

## 3.1 Commandes

| Topic | Rôle |
|---|---|
| `bioguard/cmd/auth/login` | authentification de secours côté backend MQTT |
| `bioguard/cmd/users/list` | récupération de la liste des utilisateurs |
| `bioguard/cmd/users/enroll` | création d'utilisateur avec enrôlement biométrique |
| `bioguard/cmd/users/update` | mise à jour d'un utilisateur existant |
| `bioguard/cmd/users/delete` | suppression d'un utilisateur |
| `bioguard/cmd/access/scan` | lancement d'un scan d'identification |
| `bioguard/cmd/camera/preview` | ouverture / fermeture de la prévisualisation caméra |
| `bioguard/cmd/access/logs` | récupération des événements d'accès |
| `bioguard/cmd/audit/list` | récupération des audits |
| `bioguard/cmd/settings/update` | application de réglages matériels distants |
| `bioguard/cmd/ping` | test de présence du backend |

## 3.2 Publications backend

| Topic | Rôle |
|---|---|
| `bioguard/status` | état général du backend |
| `bioguard/telemetry` | télémétrie complète du dispositif |
| `bioguard/events` | publication des événements d'accès |

## 3.3 Réponses

Le backend répond notamment sur :

- `bioguard/res/auth/login/<client_id>`
- `bioguard/res/users/list/<client_id>`
- `bioguard/res/users/enroll/<client_id>`
- `bioguard/res/users/update/<client_id>`
- `bioguard/res/users/delete/<client_id>`
- `bioguard/res/access/scan/<client_id>`
- `bioguard/res/camera/preview/<client_id>`
- `bioguard/res/access/logs/<client_id>`
- `bioguard/res/audit/list/<client_id>`
- `bioguard/res/settings/update/<client_id>`

## 4. Convention de payload

Chaque requête envoyée par le mobile injecte automatiquement :

```json
{
  "client_id": "mobile-xxxxxxxx"
}
```

Ce `client_id` permet au backend de répondre sur le bon topic corrélé.

## 5. Payloads de commande

## 5.1 Auth backend MQTT

Cette commande existe, mais elle ne remplace pas l'authentification Firebase utilisée pour l'application.

Exemple :

```json
{
  "client_id": "mobile-a1b2c3d4",
  "username": "admin@bioguard.local",
  "password": "Admin1234!"
}
```

## 5.2 Liste des utilisateurs

Exemple :

```json
{
  "client_id": "mobile-a1b2c3d4"
}
```

## 5.3 Enrôlement utilisateur

Payload typique envoyé par le mobile :

```json
{
  "client_id": "mobile-a1b2c3d4",
  "username": "Alice Martin",
  "password": "Temp1234!",
  "role": "operator",
  "email": "alice@example.com",
  "department": "Security",
  "access_groups": ["main", "lab"],
  "notes": "Utilisateur de démonstration",
  "images": []
}
```

En mode édition, le mobile n'utilise pas ce topic mais `users/update`.

## 5.4 Mise à jour utilisateur

Exemple :

```json
{
  "client_id": "mobile-a1b2c3d4",
  "user_id": "BG-USER-0001",
  "username": "Alice Martin",
  "role": "admin",
  "department": "IT",
  "email": "alice@example.com"
}
```

## 5.5 Suppression utilisateur

Exemple :

```json
{
  "client_id": "mobile-a1b2c3d4",
  "user_id": "BG-USER-0001"
}
```

## 5.6 Scan d'accès

Deux modes sont possibles.

### Scan avec utilisateur revendiqué

```json
{
  "client_id": "mobile-a1b2c3d4",
  "user_id": "BG-USER-0001"
}
```

### Scan sans utilisateur revendiqué

```json
{
  "client_id": "mobile-a1b2c3d4"
}
```

Dans ce cas, le backend recherche le meilleur candidat parmi les profils stockés.

## 5.7 Prévisualisation caméra

### Démarrer

```json
{
  "client_id": "mobile-a1b2c3d4",
  "action": "start",
  "mode": "scan",
  "user_id": "BG-USER-0001"
}
```

Le champ `user_id` est optionnel.

### Arrêter

```json
{
  "client_id": "mobile-a1b2c3d4",
  "action": "stop"
}
```

## 5.8 Lecture des logs d'accès

```json
{
  "client_id": "mobile-a1b2c3d4"
}
```

## 5.9 Lecture des audits

```json
{
  "client_id": "mobile-a1b2c3d4"
}
```

## 5.10 Réglages système

Exemple complet :

```json
{
  "client_id": "mobile-a1b2c3d4",
  "auto_light_enabled": true,
  "assist_lights_on": false,
  "green_led_on": false,
  "red_led_on": false,
  "dark_ratio": 1.25,
  "lcd_line1": "BioGuard",
  "lcd_line2": "Scan pret",
  "buzzer_test": false
}
```

Tous les champs sont optionnels. Le backend n'applique que ceux présents.

## 6. Réponses principales

## 6.1 Réponse de login backend

Succès :

```json
{
  "status": "success",
  "user": {
    "id": "admin-001",
    "username": "admin@bioguard.local",
    "role": "admin",
    "department": "Security"
  }
}
```

Échec :

```json
{
  "status": "fail",
  "error": "Invalid credentials"
}
```

## 6.2 Réponse de liste des utilisateurs

Le backend renvoie directement un tableau JSON.

Exemple :

```json
[
  {
    "id": "BG-USER-0001",
    "username": "Alice Martin",
    "email": "alice@example.com",
    "role": "operator",
    "department": "Security",
    "created_at": "2026-03-26T10:12:00Z",
    "has_biometrics": 1
  }
]
```

## 6.3 Réponse d'enrôlement

Exemple représentatif :

```json
{
  "status": "success",
  "user_id": "BG-USER-0001",
  "username": "Alice Martin",
  "biometric_key": "sha256...",
  "sample_count": 6,
  "profile_modalities": ["palmcode", "hand_pattern", "hand_geometry", "finger_geometry"],
  "profile": {},
  "preview_paths": ["..."],
  "processed_image_paths": ["..."],
  "telemetry": {}
}
```

## 6.4 Réponse de mise à jour utilisateur

```json
{
  "status": "success",
  "user": {
    "id": "BG-USER-0001",
    "username": "Alice Martin",
    "email": "alice@example.com",
    "role": "admin",
    "department": "IT",
    "created_at": "2026-03-26T10:12:00Z"
  }
}
```

## 6.5 Réponse de suppression utilisateur

```json
{
  "status": "success",
  "deleted_user_id": "BG-USER-0001"
}
```

## 6.6 Réponse de scan

Exemple de succès :

```json
{
  "status": "success",
  "result": "GRANTED",
  "user_id": "BG-USER-0001",
  "username": "Alice Martin",
  "biometric_key": "sha256...",
  "score": 0.81,
  "threshold": 0.45,
  "quality_gate_passed": true,
  "quality_reason": "",
  "quality": {},
  "components": {
    "palmcode": 0.81,
    "geometry": 0.92
  },
  "best_candidate": {
    "user_id": "BG-USER-0001",
    "username": "Alice Martin",
    "score": 0.81
  },
  "processed_image_path": "...",
  "preview_path": "...",
  "event": {}
}
```

Exemple d'échec :

```json
{
  "status": "fail",
  "reason": "NO_MATCH_FOUND",
  "best_candidate": {
    "user_id": "BG-USER-0003",
    "username": "Test User",
    "score": 0.31
  }
}
```

## 6.7 Réponse de preview caméra

```json
{
  "status": "success",
  "action": "start",
  "mode": "scan",
  "telemetry": {}
}
```

## 6.8 Réponse de réglages

```json
{
  "status": "success",
  "settings": {
    "auto_light_enabled": true,
    "assist_lights_on": false,
    "green_led_on": false,
    "red_led_on": false,
    "dark_ratio": 1.25,
    "lcd": {
      "line1": "BioGuard",
      "line2": "Scan pret"
    },
    "buzzer": {
      "pin": 23,
      "last_action": "OFF"
    }
  },
  "telemetry": {}
}
```

## 7. Publications backend

## 7.1 Statut

Le topic `bioguard/status` diffuse l'état général du backend.

Exemple :

```json
{
  "status": "ONLINE",
  "device_id": "rpi-entry-01",
  "app": "BioGuard Access",
  "local_ip": "192.168.1.10",
  "mqtt_port": 1883,
  "mqtt_ws_port": 9090,
  "timestamp": "2026-03-26T15:42:01.000000+00:00"
}
```

Des champs additionnels peuvent être ajoutés selon la phase, par exemple :

- `phase`
- `sample_index`
- `sample_count`
- `enrolled_user_id`

## 7.2 Télémétrie

Le topic `bioguard/telemetry` transporte l'état complet du système.

Structure typique :

```json
{
  "captured_at": "2026-03-26T15:42:03.000000+00:00",
  "device_id": "rpi-entry-01",
  "light_sensor": {
    "value": 180.0,
    "baseline": 180.0,
    "dark_ratio": 1.25,
    "dark_threshold": 225.0,
    "is_dark": false
  },
  "lighting": {
    "auto_enabled": true,
    "manual_enabled": false,
    "assist_lights_on": false,
    "green_led_on": false,
    "red_led_on": false
  },
  "buzzer": {
    "pin": 23,
    "last_action": "OFF"
  },
  "lcd": {
    "address": "0x27",
    "line1": "BioGuard",
    "line2": "Scan pret",
    "enabled": true
  },
  "camera": {
    "available": true,
    "mock_mode": false,
    "width": 640,
    "height": 480,
    "frame_duration_us": 40000,
    "contrast": 1.25,
    "sharpness": 1.6
  }
}
```

Pendant une preview ou après un traitement, le bloc `camera` peut aussi contenir :

- `preview_jpeg_base64`
- `processed_jpeg_base64`

## 7.3 Événements d'accès

Le topic `bioguard/events` publie les événements de décision.

Structure typique :

```json
{
  "id": "uuid",
  "user_id": "BG-USER-0001",
  "username": "Alice Martin",
  "status": "GRANTED",
  "score": 0.81,
  "reason": "MATCH",
  "method": "multimodal_scan",
  "modalities": {},
  "device_id": "rpi-entry-01",
  "timestamp": "2026-03-26T15:43:12.000000+00:00"
}
```

## 8. Structures persistées en SQLite

## 8.1 Tables

| Table | Rôle |
|---|---|
| `users` | profils utilisateurs locaux |
| `biometric_profiles` | profils biométriques JSON |
| `access_events` | historique local des accès |
| `audit_logs` | journal des opérations administratives |
| `device_state` | cache de télémétrie et états appliqués |

## 8.2 Données importantes

### Utilisateur

Champs principaux :

- `id`
- `username`
- `email`
- `password_hash`
- `role`
- `department`
- `firebase_uid`
- `created_at`

### Profil biométrique

Le profil biométrique est stocké comme JSON. Il contient au minimum :

- `schema_version`
- `sensor`
- `modalities`
- `palmprint`
- `surface_texture`
- `finger_geometry`
- `hand_pattern`
- `biometric_key`

### Événement d'accès

Champs principaux :

- `id`
- `user_id`
- `username`
- `status`
- `score`
- `reason`
- `method`
- `modalities`
- `timestamp`

## 9. Structures Firestore

## 9.1 Collections utilisées

- `users`
- `biometric_profiles`
- `access_events`
- `device_telemetry`
- `mobile_user_preferences`

## 9.2 Rôle par collection

### `users`

Contient les profils administratifs et métiers visibles côté mobile.

### `biometric_profiles`

Contient les profils biométriques synchronisés.

### `access_events`

Conserve les événements d'accès pour consultation distante.

### `device_telemetry`

Conserve le dernier état du dispositif par `device_id`.

### `mobile_user_preferences`

Conserve les préférences d'affichage et de comportement du mobile.

## 10. Stratégie de cohérence des données

Le système suit une logique pragmatique.

- SQLite est la vérité locale du backend en temps réel.
- Firestore est une projection distante utile pour consultation et synchronisation.
- Le mobile essaie de consommer MQTT en priorité.
- En cas d'indisponibilité du backend MQTT, certaines données peuvent être relues depuis Firestore.

## 11. Remarques importantes

### 11.1 Authentification applicative vs authentification MQTT

Le mobile utilise principalement Firebase Authentication pour connecter l'utilisateur humain.

Le login MQTT backend existe surtout comme fonctionnalité complémentaire ou de démonstration. Il ne doit pas être confondu avec la session Firebase.

### 11.2 Images base64

Les previews et images traitées sont encapsulées dans la télémétrie sous forme base64. Elles sont pratiques pour la démo mais :

- augmentent la taille des messages
- ne doivent pas être considérées comme un format optimisé pour de gros déploiements

### 11.3 Biométrie et volumétrie

Les profils biométriques sont stockés en JSON. C'est pratique pour un prototype, mais ce n'est pas un format optimisé pour un système industriel à grande échelle.

## 12. Conclusion

Le contrat MQTT et les structures de données de BioGuard Access sont volontairement simples, lisibles et cohérents avec un prototype IoT de démonstration. Ils permettent :

- une communication temps réel compréhensible
- une journalisation exploitable
- une synchronisation cloud minimale mais utile
- une séparation claire entre pilotage, supervision et persistance
