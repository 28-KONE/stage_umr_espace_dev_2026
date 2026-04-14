# stage_umr_espace_dev_2026

## Apprentissage profond auto-supervisé pour la cartographie multi-échelle des mangroves à partir d’images Pléiades et Sentinel

Dépôt privé de développement créé dans le cadre de mon stage de M2 2026.
Ce projet s’inscrit dans les programmes de recherche :

* **ESA Coastal Blue Carbon** (2024–2026)
* **ANIMALS** – *Artificial iNtellIgence for the mapping of MAngrove at Large Scale* (CNES, 2025–2027)

---

# Objectif

Ce stage vise à développer des méthodes d’**intelligence artificielle auto-supervisée** pour cartographier les mangroves à différentes échelles spatiales à partir d’images satellites :

* **Pléiades** (Très Haute Résolution Spatiale – 50 cm)
* **Sentinel-1 / Sentinel-2** (Haute Résolution Spatiale – 10 m)

L’objectif principal est de transférer l’information fine issue des images Pléiades vers les images Sentinel afin de produire des cartographies robustes à large échelle :

* Étendue des mangroves
* Types d’habitats forestiers
* Biomasse aérienne
* Stocks de carbone

---

# Contexte

Les mangroves jouent un rôle majeur dans :

* la protection des littoraux
* la biodiversité
* le stockage du carbone bleu
* l’atténuation du changement climatique

La zone d’étude principale concerne :

* **Guyane française**
* Côte sous influence amazonienne :

  * Brésil
  * Suriname
  * Guyana

---

# Approche IA

Le projet repose sur des approches de **self-supervised learning** appliquées à la télédétection.

Modèles envisagés :

* **CROMA** (Foundation model)
* autres architectures selon expérimentation

Objectif :

> entraîner un modèle à reproduire des annotations fines générées à partir d’images Pléiades, en utilisant uniquement des images Sentinel à grande échelle.

---

# Structure du repository

```text id="repo01"
stage_umr_espace_dev_2026/
│
├── src/
|   │
|   ├── README.md
|   │
|   ├── data_acquisition/
|   │   ├── README.md
|   │   ├── generate_bboxes_mangrove.py
|   │   ├── download_sentinel1.py
|   │   ├── download_sentinel2.py
|   │   └── run_downloads.py
|   │
|   ├── preprocessing/
|   ├── models/
|   ├── training/
|   ├── inference/
|   └── utils/
│
├── notebooks/          # Exploration / prototypes
├── docs/               # Documentation technique
├── tests/              # Tests unitaires
├── configs/            # Fichiers YAML / paramètres
├── outputs/            # Résultats, cartes, modèles
├── environment.yml     # Environnement Conda
├── requirements.txt
└── README.md
```
##  Documentation

Le dossier `docs/` contient la documentation technique du projet.

* `Prise_en_main_et_téléchargement_avec_EODAG.pdf`
  Présentation de la procédure de téléchargement Sentinel-1 / Sentinel-2 avec EODAG.

---

# Missions principales

*  État de l’art sur les méthodes IA de passage à l’échelle
*  Implémentation d’un modèle auto-supervisé
*  Construction de la base de données image
*  Entraînement spécifique aux mangroves
*  Production de cartes de texture (Pléiades)
*  Cartographie grande échelle (Sentinel)
*  Estimation biomasse / carbone
*  Documentation technique complète
*  Rapport de stage

---

#  Installation

## 1. Cloner le dépôt

```bash id="git01"
git clone <url-du-repo>
cd stage_umr_espace_dev_2026
```

## 2. Installer les dépendances

```bash id="git02"
pip install -r requirements.txt
```

ou

```bash id="git03"
conda env create -f environment.yml
conda activate mangrove-ai
```

---

# Exemple d’utilisation

```bash id="run01"
python src/training/train.py
```

```bash id="run02"
python src/inference/predict.py
```

---

# Technologies utilisées

* Python
* PyTorch
* Rasterio / GDAL
* GeoPandas
* NumPy / Pandas
* Jupyter
* Git / GitHub

---

# Résultats attendus

* Cartes annuelles des habitats forestiers de mangrove
* Détection automatique des mangroves
* Biomasse aérienne estimée
* Stocks de carbone spatialement distribués
* Méthode IA reproductible

---

# Bonnes pratiques

* Ne pas versionner les données sensibles ou volumineuses
* Utiliser `.gitignore`
* Commits réguliers et explicites
* Documentation continue

---

# Auteur

28-KONE
Master 2 Machine Learning for Artificial Intelligence – Stage 2026
UMR Espace Dev - Université de Montpellier

---

# Licence

Usage académique / recherche interne.
