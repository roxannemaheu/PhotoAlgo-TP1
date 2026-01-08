# Photographie Algorithmique GIF4105/7105 -- Code de démarrage TP1

Ce dépôt contient le code de démarrage pour le Travail Pratique #1 du cours de photographie algorithmique (GIF-4105/7105) à l'Université Laval.  
L'énoncé du TP est disponible ici : [https://wcours.gel.ulaval.ca/GIF4105/tps/tp1/]([https://wcours.gel.ulaval.ca/GIF4105/tps/tp1/)

## Installation

### Prérequis : Python

Si vous n'avez pas encore Python installé sur votre système :

1. **Télécharger Python** :
   - Visitez [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Téléchargez une version de Python récente, entre 3.11 et 3.13, pour votre système d'exploitation
   - **Important** : Lors de l'installation sur Windows, cochez l'option "Add Python to PATH" pour pouvoir utiliser Python depuis la ligne de commande
   - Exécutez l'installateur et suivez les instructions

2. **Vérifier l'installation** :
   ```bash
   python --version
   ```

### Création de l'environnement Python

1. **Créer un nouvel environnement virtuel** :
   ```bash
   python -m venv venv
   ```

2. **Activer l'environnement** :
   - Sur Windows :
     ```bash
     venv\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```bash
     source venv/bin/activate
     ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

### Utilisation

Pour utiliser l'environnement dans une nouvelle session terminal :
- Sur Windows :
  ```bash
  venv\Scripts\activate
  ```
- Sur macOS/Linux :
  ```bash
  source venv/bin/activate
  ```

Pour désactiver l'environnement :
```bash
deactivate
```
