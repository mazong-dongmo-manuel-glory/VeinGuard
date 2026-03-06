# VeinGuard 🔐

**Système de contrôle d'accès biométrique intelligent basé sur la reconnaissance de veines.**

VeinGuard est un projet IoT intégrant une application mobile React Native, un backend Node.js, et un microcontrôleur ESP32 pour sécuriser l'accès à des zones restreintes via la biométrie veineuse.

---

## Structure du projet

```
VeinGuard/
├── Mobile/          # Application mobile React Native (Expo)
├── iot/             # Algorithme de reconnaissance veineuse (Python / ESP32)
└── Maquettes/       # Maquettes UI de l'application
```

---

## Mobile – Application React Native

Application mobile permettant de :
- S'authentifier (compte utilisateur)
- Simuler un scan biométrique de veine
- Communiquer avec l'ESP32 via MQTT
- Consulter l'historique des accès
- Gérer les utilisateurs autorisés (CRUD)

### Stack
- **React Native** (Expo)
- **MQTT** – communication avec l'ESP32
- **AsyncStorage** – stockage local des paramètres

### Lancer l'application

```bash
cd Mobile
npm install
npm start
```

---

## IoT – Algorithme PBBM (Python)

Implémentation Python du **Personalized Best Bit Map (PBBM)** pour la reconnaissance biométrique veineuse.

### Fichiers principaux
| Fichier | Description |
|---|---|
| `pbbm.py` | Algorithme principal PBBM |
| `batch_eval_pbbm.py` | Évaluation batch du modèle |

> ⚠️ **Dataset non inclus** – Les images biométriques (`data/`) ne sont pas versionnées pour des raisons de taille et de confidentialité.

### Lancer l'évaluation

```bash
cd iot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # à créer selon vos dépendances
python batch_eval_pbbm.py
```

---

## Architecture du système

```
Utilisateur
    │
Application Mobile (React Native)
    │
API Backend (Node.js / Express)
    │
MQTT Broker
    │
ESP32
    │
Servo moteur → Ouverture de porte
```

---

## Matériel requis

| Composant | Prix |
|---|---|
| ESP32 | ~7$ |
| Capteur IR | ~20$ |
| Servo moteur | ~5$ |
| Boîtier | ~10$ |
| **Total** | **~42$** |

---

## Contributeurs

Projet réalisé dans le cadre d'un cours d'IoT.
