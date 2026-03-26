# Installation Et Execution

## 1. Objectif de ce guide

Ce document décrit une procédure d'installation réaliste pour exécuter le projet dans un contexte de développement, de démonstration ou de soutenance.

Le système complet implique trois briques :

- un broker MQTT
- le backend IoT Python sur Raspberry Pi
- l'application mobile Expo

## 2. Pré-requis généraux

## 2.1 Matériel minimal

- un Raspberry Pi
- une caméra Pi compatible ou un environnement de mock
- un capteur de lumière si démonstration matérielle complète
- un LCD I2C, un buzzer et des LED si démonstration physique complète
- un téléphone ou un simulateur pour l'application mobile

## 2.2 Logiciels minimaux

- Python 3 sur le Raspberry Pi
- `pip`
- un broker MQTT, par exemple Mosquitto
- Node.js et npm pour le mobile
- Expo via `npx expo`

## 2.3 Réseau

Le mobile doit connaître :

- l'hôte du broker ou du Raspberry Pi
- le port MQTT WebSocket utilisé par le mobile
- le port MQTT TCP utilisé par le backend

## 3. Installation du broker MQTT

Le projet suppose l'existence d'un broker accessible en :

- TCP pour le Raspberry Pi
- WebSocket pour l'application mobile

## 3.1 Exemple avec Mosquitto sur Raspberry Pi

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

## 3.2 Création d'un utilisateur MQTT

Le projet utilise par défaut un compte de démonstration :

- utilisateur : `admin`
- mot de passe : `admin1234`

Commande de création :

```bash
sudo mosquitto_passwd -b /etc/mosquitto/passwd admin admin1234
sudo systemctl restart mosquitto
```

## 3.3 Vérification rapide du broker

Depuis le Raspberry Pi :

```bash
mosquitto_pub -h localhost -p 1883 -u admin -P admin1234 -t test -m hello
mosquitto_sub -h localhost -p 1883 -u admin -P admin1234 -t test
```

## 3.4 Important pour le mobile

L'application mobile se connecte en WebSocket. Le broker doit donc exposer un port WebSocket, par exemple `9090`.

Si Mosquitto n'est pas déjà configuré pour cela, il faut ajouter un listener WebSocket dans sa configuration.

Exemple simplifié :

```conf
listener 1883
protocol mqtt

listener 9090
protocol websockets

allow_anonymous false
password_file /etc/mosquitto/passwd
```

Puis :

```bash
sudo systemctl restart mosquitto
```

## 4. Installation du backend IoT

## 4.1 Aller dans le dossier backend

```bash
cd iot
```

## 4.2 Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4.3 Installer les dépendances

```bash
pip install -r requirements.txt
```

Les dépendances importantes incluent :

- `paho-mqtt`
- `opencv-python-headless`
- `numpy`
- `firebase-admin`
- `gpiozero`
- `picamera2`
- `RPLCD`
- `Werkzeug`

## 4.4 Variables d'environnement utiles

Pour un lancement local simple :

```bash
export VG_MOCK_MODE=1
export VG_MQTT_BROKER=localhost
export VG_MQTT_PORT=1883
export VG_MQTT_WS_PORT=9090
export VG_MQTT_USERNAME=admin
export VG_MQTT_PASSWORD=admin1234
export VG_FIREBASE_ENABLED=0
```

### Signification rapide

| Variable | Rôle |
|---|---|
| `VG_MOCK_MODE` | active des fallbacks si le matériel réel n'est pas branché |
| `VG_MQTT_BROKER` | hôte du broker MQTT |
| `VG_MQTT_PORT` | port MQTT TCP pour le Pi |
| `VG_MQTT_WS_PORT` | port WebSocket MQTT, utilisé surtout par le mobile |
| `VG_MQTT_USERNAME` | identifiant du broker |
| `VG_MQTT_PASSWORD` | mot de passe du broker |
| `VG_FIREBASE_ENABLED` | active ou non la couche Firebase |

## 4.5 Paramètres biométriques utiles

Le backend intègre aussi plusieurs variables pour la biométrie. Les plus importantes pour le nouveau pipeline PalmCode sont :

- `VG_PALM_CODE_ORIENTATIONS`
- `VG_PALM_CODE_RING_COUNT`
- `VG_PALM_CODE_RING_OVERLAP`
- `VG_PALMCODE_MATCH_THRESHOLD`
- `VG_PALM_CODE_SIZE`

Dans la plupart des cas, les valeurs par défaut suffisent pour la démo initiale.

## 5. Configuration Firebase côté backend

Firebase est facultatif. Si vous voulez l'activer :

```bash
export VG_FIREBASE_ENABLED=1
export VG_FIREBASE_CREDENTIALS=/chemin/vers/firebase-service-account.json
export VG_FIREBASE_PROJECT_ID=veinguard-d127f
export VG_FIREBASE_STORAGE_BUCKET=veinguard-d127f.firebasestorage.app
```

Conditions nécessaires :

- le fichier de service account doit exister
- `firebase-admin` doit être installé
- le projet Firebase doit contenir les bonnes collections

Si Firebase n'est pas disponible, le backend continue à fonctionner avec SQLite.

## 6. Lancement du backend

Depuis le dossier `iot/` :

```bash
python mqtt_gateway.py
```

Ou explicitement avec Python 3 :

```bash
python3 mqtt_gateway.py
```

## 6.1 Ce que vous devez observer au démarrage

- initialisation de la base SQLite
- séquence de boot sur le matériel
- connexion au broker MQTT
- publication du statut `ONLINE`
- émission régulière de télémétrie

## 6.2 Emplacement des données backend

Le backend crée et utilise :

- `iot/data/`
- `iot/data/captures/`
- `iot/veinguard.db`

## 7. Installation de l'application mobile

## 7.1 Aller dans le dossier mobile

```bash
cd Mobile
```

## 7.2 Installer les dépendances

```bash
npm install
```

Le projet mobile repose sur :

- Expo
- React Native
- Firebase JS SDK
- MQTT over WebSocket
- Zustand
- React Navigation

## 7.3 Vérifier la configuration mobile

Le fichier principal de configuration est :

- [Mobile/config.js](/Users/mazong/Documents/GitHub/VeinGuard/Mobile/config.js)

Il contient notamment :

- l'hôte MQTT par défaut
- les ports MQTT
- les identifiants MQTT
- les topics
- la configuration Firebase côté mobile

## 7.4 Configuration Firebase mobile

Le mobile utilise `FIREBASE_CONFIG` via `Mobile/config.js`.

Produits Firebase attendus :

- Authentication
- Firestore

Si vous voulez éviter de hardcoder les secrets, utilisez les variables `EXPO_PUBLIC_FIREBASE_*`.

## 8. Lancement du mobile

```bash
cd Mobile
npm start
```

Puis :

- `a` pour Android
- `i` pour iOS si environnement compatible
- ou scan du QR code avec Expo Go

Autres commandes utiles :

```bash
npm run android
npm run ios
npm run web
```

## 9. Ordre de démarrage conseillé

Pour une démonstration stable, lancer dans cet ordre :

1. le broker MQTT
2. le backend IoT sur le Pi
3. l'application mobile
4. la connexion Firebase si utilisée
5. la vérification du preview caméra

## 10. Vérification fonctionnelle minimale

Une fois le système démarré :

1. vérifier que le backend publie `ONLINE`
2. vérifier que la télémétrie arrive dans le mobile
3. vérifier que la configuration de l'hôte MQTT est correcte dans le mobile
4. se connecter via Firebase sur l'application
5. ouvrir l'écran des paramètres et vérifier la télémétrie
6. ouvrir un preview caméra
7. enrôler un utilisateur
8. effectuer un scan
9. consulter l'historique et les audits

## 11. Vérification rapide avec un Pi réel

Exemple de séquence complète :

```bash
cd ~/Desktop/VeinGuard/iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export VG_MOCK_MODE=0
export VG_MQTT_BROKER=localhost
export VG_MQTT_PORT=1883
export VG_MQTT_WS_PORT=9090
export VG_MQTT_USERNAME=admin
export VG_MQTT_PASSWORD=admin1234
export VG_FIREBASE_ENABLED=0

python3 mqtt_gateway.py
```

## 12. Dépannage courant

## 12.1 Le mobile ne voit pas le backend

Vérifier :

- l'adresse IP configurée dans l'écran `Login` ou `Paramètres`
- le port WebSocket MQTT
- que le broker expose bien un listener WebSocket
- que le téléphone est sur le même réseau

## 12.2 Le backend ne se connecte pas au broker

Vérifier :

- `VG_MQTT_BROKER`
- `VG_MQTT_PORT`
- le couple `VG_MQTT_USERNAME` / `VG_MQTT_PASSWORD`
- la configuration ACL du broker

## 12.3 Firebase ne fonctionne pas côté Pi

Vérifier :

- `VG_FIREBASE_ENABLED=1`
- le chemin du service account
- l'installation de `firebase-admin`
- le projet Firebase associé

## 12.4 La caméra ne fonctionne pas

Vérifier :

- la présence de `picamera2`
- la compatibilité de la caméra
- les permissions système
- ou, pour le développement, utiliser `VG_MOCK_MODE=1`

## 12.5 Le scan biométrique est instable

Vérifier :

- l'éclairage ambiant
- l'état des LED d'appoint
- la position de la main
- la netteté de l'image
- le seuil `VG_PALMCODE_MATCH_THRESHOLD`

## 12.6 Le mobile démarre mais Firebase échoue

Vérifier :

- la configuration dans `Mobile/config.js`
- l'activation de Firebase Authentication
- les règles Firestore
- la connectivité Internet

## 13. Conseils pour la soutenance

Pour éviter les mauvaises surprises pendant une démo :

- préparer un mode `VG_MOCK_MODE=1` en secours
- vérifier le broker WebSocket à l'avance
- garder les identifiants MQTT de démonstration cohérents entre Pi et mobile
- valider la preview caméra avant de montrer l'enrôlement
- préparer au moins un utilisateur déjà enrôlé

## 14. Résumé opérationnel

En version la plus simple :

### Backend

```bash
cd iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export VG_MOCK_MODE=1
export VG_MQTT_BROKER=localhost
export VG_MQTT_USERNAME=admin
export VG_MQTT_PASSWORD=admin1234
python3 mqtt_gateway.py
```

### Mobile

```bash
cd Mobile
npm install
npm start
```

## 15. Conclusion

L'installation du projet reste raisonnable pour un prototype académique, à condition de traiter correctement :

- le broker MQTT
- la configuration réseau
- la présence ou non du matériel réel
- la couche Firebase

Le backend peut fonctionner seul localement, et le mobile vient ensuite compléter le système par la supervision et l'administration.
