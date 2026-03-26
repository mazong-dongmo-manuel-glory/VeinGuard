# MQTT Et Donnees

## Topics MQTT

### Commandes

- `bioguard/cmd/auth/login`
- `bioguard/cmd/users/list`
- `bioguard/cmd/users/enroll`
- `bioguard/cmd/users/update`
- `bioguard/cmd/users/delete`
- `bioguard/cmd/access/scan`
- `bioguard/cmd/access/logs`
- `bioguard/cmd/audit/list`
- `bioguard/cmd/settings/update`
- `bioguard/cmd/ping`

### Publications

- `bioguard/status`
- `bioguard/telemetry`
- `bioguard/events`

### Réponses

Le backend répond sur des topics de type :

- `bioguard/res/<commande>/<client_id>`

Exemples :

- `bioguard/res/users/list/mobile-xxxx`
- `bioguard/res/settings/update/mobile-xxxx`

## Types de données

### Télémétrie

La télémétrie comprend notamment :

- horodatage
- `device_id`
- état du capteur de lumière
- état des LED
- état du buzzer
- contenu affiché sur le LCD
- état de la caméra

### Utilisateur

Un utilisateur contient :

- `id`
- `username`
- `email`
- `role`
- `department`
- `created_at`

### Événement d'accès

Un événement contient :

- `id`
- `user_id`
- `username`
- `status`
- `score`
- `reason`
- `method`
- `modalities`
- `device_id`
- `timestamp`

## Firestore

Collections utilisées :

- `users`
- `biometric_profiles`
- `access_events`
- `device_telemetry`
- `mobile_user_preferences`

## SQLite

Tables locales :

- `users`
- `biometric_profiles`
- `access_events`
- `audit_logs`
- `device_state`

## Stratégie de stockage

- SQLite : continuité locale, cache edge, fonctionnement même sans Internet
- Firebase : consultation distante, auth, préférences utilisateur, visibilité mobile
- MQTT : action temps réel et retour immédiat
