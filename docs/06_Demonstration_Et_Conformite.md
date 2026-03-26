# Demonstration Et Conformite

## Script de demonstration

### Pitch

1. Présenter le problème réel de contrôle d'accès.
2. Montrer la valeur commerciale du prototype.
3. Expliquer que le Raspberry Pi agit comme passerelle biométrique.

### Démonstration technique

1. Ouvrir l'application mobile.
2. Se connecter avec Firebase.
3. Montrer les préférences utilisateur.
4. Ajouter un utilisateur.
5. Modifier le même utilisateur.
6. Lancer un scan.
7. Montrer la réaction du LCD, des LED et du buzzer.
8. Consulter l'historique d'accès.
9. Consulter les journaux d'audit.
10. Supprimer l'utilisateur.

## Conformite au sujet

### Points couverts

- électronique embarquée
- connectivité mobile
- MQTT
- backend structuré
- gestion d'erreurs de base
- mini plan d'affaires
- application cellulaire

### Points partiellement couverts

- impression 3D : à documenter avec des fichiers ou visuels de boîtier si vous voulez une preuve complète
- code commenté : structure claire, mais commentaires encore limités

### Point non couvert

- minimum 5 capteurs différents

Etat réel du prototype :

- capteurs présents : caméra, capteur de lumière
- actionneurs présents : LCD, buzzer, LED rouge, LED verte, deux LED d'éclairage

## Recommandation pour le jury / prof

Présenter honnêtement le projet comme :

- un prototype fonctionnel solide sur MQTT + mobile + Firebase
- un backend bien structuré
- une démo live réaliste

Mais ne pas prétendre satisfaire l'exigence des 5 capteurs si le matériel n'existe pas physiquement.
