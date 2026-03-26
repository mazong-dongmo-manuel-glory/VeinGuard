# Demonstration Et Conformite

## 1. Objectif de ce document

Ce document sert à préparer :

- une démonstration orale
- une soutenance technique
- une lecture honnête de la conformité du prototype par rapport au sujet ou au cahier des charges

Il ne cherche pas à enjoliver le projet. Il identifie clairement :

- ce qui est bien couvert
- ce qui est partiellement couvert
- ce qui ne l'est pas encore

## 2. Pitch de présentation recommandé

Voici une trame courte, crédible et cohérente pour introduire le projet.

> BioGuard Access est un prototype de contrôle d'accès intelligent construit autour d'un Raspberry Pi. Le système combine un backend IoT, une application mobile et une biométrie locale de paume pour démontrer un contrôle d'accès embarqué, supervisable à distance, mais capable de fonctionner localement grâce à SQLite.

Points à mettre en avant immédiatement :

- le Pi prend la décision localement
- le mobile sert à administrer et superviser
- MQTT assure le temps réel
- Firebase sert à l'authentification et à la synchronisation
- la biométrie est explicable et non basée sur un gros modèle opaque

## 3. Démonstration conseillée en direct

## 3.1 Préparation avant passage

Avant de commencer :

- démarrer le broker MQTT
- démarrer le backend IoT
- vérifier que l'application mobile est connectée
- vérifier la preview caméra
- avoir au moins un utilisateur déjà enrôlé si vous voulez garantir une démo fluide

## 3.2 Déroulé recommandé

### Étape 1. Présentation rapide du contexte

Expliquer en 20 à 30 secondes :

- le besoin de contrôle d'accès
- la séparation entre mobile et Raspberry Pi
- le rôle de la biométrie de paume

### Étape 2. Montrer l'application mobile

Afficher :

- l'écran de connexion
- la configuration du serveur
- la connexion Firebase

Insister sur le fait que le mobile n'est pas juste une maquette, mais qu'il pilote réellement le backend via MQTT.

### Étape 3. Montrer l'état du système

Depuis `Dashboard` ou `SystemSetting`, montrer :

- l'état `ONLINE`
- la télémétrie du capteur de lumière
- la disponibilité de la caméra
- le contenu LCD

### Étape 4. Ouvrir une preview caméra

Depuis l'écran de scan ou d'enrôlement :

- démarrer la preview
- montrer que le flux affiché provient bien du Pi
- rappeler que l'image est transmise via la télémétrie MQTT

### Étape 5. Enrôler un utilisateur

Montrer :

- le formulaire utilisateur
- les informations métier
- la séquence multi-capture
- le retour de succès avec identifiant et profil

### Étape 6. Modifier puis supprimer un utilisateur

Cette partie démontre :

- l'administration distante
- la cohérence des données
- la traçabilité via l'audit

### Étape 7. Lancer un scan d'identification

Montrer :

- la preview
- l'animation de scan
- la réaction des actionneurs
- la décision retournée
- le score et les détails disponibles

### Étape 8. Consulter l'historique et les audits

Montrer :

- les événements d'accès
- les logs d'audit
- la persistance et la consultation sur mobile

### Étape 9. Expliquer l'architecture

Finir avec un schéma ou une explication simple :

- mobile
- MQTT
- Raspberry Pi
- SQLite
- Firebase

## 4. Démonstration courte en moins de 5 minutes

Si le temps est très limité :

1. ouvrir `Dashboard`
2. montrer l'état du système
3. ouvrir la preview caméra
4. lancer un scan
5. montrer l'événement généré
6. afficher rapidement l'historique

## 5. Démonstration complète en 8 à 12 minutes

Séquence recommandée :

1. contexte et problème
2. architecture globale
3. connexion mobile
4. télémétrie
5. preview
6. enrôlement
7. scan réussi
8. scan refusé ou échec contrôlé
9. historique
10. audit
11. conformité et limites

## 6. Éléments techniques à verbaliser

Les points suivants valorisent bien le projet sans survendre.

### 6.1 Backend embarqué réel

- le Raspberry Pi pilote vraiment le matériel
- la logique n'est pas simulée côté mobile

### 6.2 Communication temps réel

- le mobile publie des commandes via MQTT
- le backend répond sur des topics corrélés par `client_id`

### 6.3 Persistance hybride

- SQLite pour la continuité locale
- Firestore pour la synchronisation et la consultation distante

### 6.4 Biométrie explicable

- ROI palmaire alignée
- filtrage Gabor
- anneaux concentriques
- vecteur PalmCode
- comparaison par similarité

## 7. Conformité au sujet

## 7.1 Points clairement couverts

Le prototype couvre bien :

- électronique embarquée
- pilotage d'actionneurs
- lecture de capteurs
- application mobile
- communication réseau temps réel
- stockage local
- stockage cloud
- architecture logicielle structurée
- démonstration d'un cas d'usage concret

## 7.2 Points couverts partiellement

### Biométrie

Le projet propose une biométrie locale cohérente, mais ce n'est pas une solution industrielle certifiée. Pour un projet académique, cela reste un point fort, à condition de le présenter comme un prototype.

### Industrialisation

Le système est bien structuré, mais il manque plusieurs briques d'un produit prêt au déploiement :

- durcissement sécurité complet
- boîtier industrialisé final
- automatisation CI/CD
- monitoring avancé

### Preuve matérielle détaillée

Si le sujet exige des preuves de câblage ou de montage, il peut être utile d'ajouter :

- schéma de câblage
- photos du montage
- vues du boîtier ou maquettes 3D

## 7.3 Point non totalement couvert

Le point le plus important à assumer est le suivant :

- le prototype ne présente pas aujourd'hui cinq capteurs différents réels

État actuel :

- capteurs réellement intégrés : caméra, capteur de lumière
- actionneurs intégrés : LCD, buzzer, LED rouge, LED verte, LED d'éclairage

Il ne faut pas présenter les actionneurs comme des capteurs.

## 8. Comment présenter honnêtement cette limite

La meilleure formulation est sobre et défendable.

Exemple :

> Le prototype couvre bien la logique IoT, la communication temps réel, la supervision mobile et un pipeline biométrique local. En revanche, sur le critère strict du nombre de capteurs physiques distincts, l'implémentation actuelle reste partielle. Nous avons privilégié une intégration complète et démontrable de quelques composants réels plutôt qu'une multiplication artificielle de capteurs peu exploités.

Cette réponse montre :

- de l'honnêteté
- une vraie maîtrise technique
- une capacité à arbitrer les choix de conception

## 9. Questions probables du jury et réponses conseillées

### Pourquoi MQTT et pas une API REST ?

Réponse conseillée :

MQTT simplifie le temps réel, correspond bien à un scénario IoT, réduit la complexité du backend et permet au mobile de piloter le Pi sans ajouter une couche HTTP supplémentaire.

### Pourquoi Firebase en plus de SQLite ?

Réponse conseillée :

SQLite garantit la continuité locale sur le Pi. Firebase sert à l'authentification mobile et à la consultation distante. Les deux répondent à des besoins différents.

### Pourquoi ne pas utiliser un modèle d'IA plus puissant ?

Réponse conseillée :

Le but était de conserver un pipeline explicable, léger et exécuté localement sur Raspberry Pi. Le PalmCode par filtrage multi-orientation et statistiques d'anneaux est plus défendable dans ce cadre.

### Le mobile pourrait-il remplacer le backend ?

Réponse conseillée :

Non, car le mobile ne pilote pas directement le matériel ni la décision locale. Il agit comme interface opérateur, alors que le Pi reste le nœud de contrôle d'accès.

## 10. Checklist avant soutenance

### Logiciel

- broker MQTT démarré
- backend lancé
- mobile lancé
- Firebase vérifié si utilisé
- preview caméra testée

### Données

- au moins un utilisateur enrôlé
- historique et audits non vides
- paramètres MQTT corrects sur le mobile

### Matériel

- LEDs fonctionnelles
- buzzer fonctionnel
- LCD fonctionnel
- caméra disponible
- éclairage d'appoint vérifié

## 11. Stratégie de secours si la démo matérielle échoue

Prévoir un plan B :

- activer `VG_MOCK_MODE=1`
- garder des captures et profils déjà générés
- montrer l'architecture, les flux et la télémétrie même si la caméra réelle pose problème

Le projet reste défendable si vous montrez :

- la structure logicielle
- la communication MQTT
- la persistance
- le mobile fonctionnel

## 12. Conclusion

BioGuard Access est très solide comme prototype de soutenance si vous le présentez pour ce qu'il est réellement :

- un système IoT complet de bout en bout
- un contrôle d'accès embarqué démontrable
- une intégration sérieuse entre matériel, backend, mobile et cloud

Son principal écart à assumer concerne le nombre de capteurs physiques distincts. Pour le reste, le projet possède une base technique crédible, structurée et démontrable.
