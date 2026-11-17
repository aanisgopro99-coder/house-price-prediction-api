# Déploiement d’un modèle de prédiction du prix des maisons

Projet pédagogique du TP Module 17 : entraîner un modèle de régression, le sauvegarder, l'exposer avec une API FastAPI, le tester et préparer son déploiement avec Docker et GitHub Actions.

## 1. Prérequis

- Python 3.11
- Git
- Docker Desktop (pour l'étape Docker)

## 2. Activer l'environnement virtuel

Dans PowerShell, depuis le dossier du projet :

    .\env\Scripts\Activate.ps1

Si PowerShell bloque l'activation, utiliser directement l'interpréteur de l'environnement :

    .\env\Scripts\python.exe --version

## 3. Installer les dépendances

    python -m pip install -r requirements-dev.txt

## 4. Générer les données et entraîner le modèle

    python -m src.generer_donnees
    python -m src.entrainer

Le jeu de données est créé dans data/maisons.csv. Le modèle et ses métadonnées sont enregistrés dans models/.

## 5. Lancer l'API

    python -m uvicorn src.main:app --reload

- Documentation interactive : http://127.0.0.1:8000/docs
- Santé du service : http://127.0.0.1:8000/health
- Informations du modèle : http://127.0.0.1:8000/model-info

Exemple de requête JSON (POST vers /predict) :

    {
      "surface": 100,
      "chambres": 3
    }

## 6. Exécuter les tests

    python -m pytest -q

## 7. Construire et lancer avec Docker

Après avoir généré le modèle :

    docker build -t house-price-api .
    docker run --rm -p 8000:8000 house-price-api

Ouvrir ensuite http://127.0.0.1:8000/docs.

## Notes de sécurité et de reproductibilité

- models/modele.pkl est un fichier pickle : ne jamais charger un pickle provenant d'une source non fiable.
- Les versions des dépendances sont fixées dans requirements.txt et requirements-dev.txt.
- Le modèle est généré localement et dans le pipeline CI à partir du script de génération de données.
- Ne jamais enregistrer de mots de passe, clés API ou secrets dans Git.
- Le déploiement cloud et la configuration de DVC seront réalisés dans les étapes suivantes du TP.