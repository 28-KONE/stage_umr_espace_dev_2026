# Dossier `src`

Ce dossier contient l’ensemble du code source du projet.

Il est organisé par grandes étapes du pipeline :

---

# Contenu

## `data_acquisition/`

Scripts de téléchargement et préparation initiale des données satellites.

Contient :

* génération des zones d’étude (BBOX)
* téléchargement Sentinel-1
* téléchargement Sentinel-2
* pipeline automatisé

## `preprocessing/`

Prétraitements futurs :

* reprojection
* mosaïques
* extraction de patches
* normalisation
* masques nuages

## `models/`

Implémentation des modèles d’intelligence artificielle :

* CROMA
* autoencodeurs
* architectures CNN / Transformers

## `training/`

Scripts d’entraînement des modèles.

## `inference/`

Prédictions et génération des cartes finales.

## `utils/`

Fonctions utilitaires communes.
