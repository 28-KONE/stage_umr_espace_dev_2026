# Data Acquisition

## Téléchargement et organisation des données Sentinel-1 / Sentinel-2

Ce module contient les scripts développés pour télécharger automatiquement les données satellitaires nécessaires à l’étude des mangroves guyanaises.

---

# Contexte

Les mangroves sont des écosystèmes dynamiques nécessitant une couverture spatiale et temporelle régulière.

Deux capteurs sont utilisés :

* **Sentinel-1** : radar, indépendant des nuages
* **Sentinel-2** : optique multispectral

---

# Scripts disponibles

## `generate_bboxes_mangrove.py`

Génère automatiquement les zones côtières (BBOX) à partir du masque mangrove 2023 avec buffer de 10 km.

## `download_sentinel1.py`

Télécharge les produits Sentinel-1 GRD.

## `download_sentinel2.py`

Télécharge les produits Sentinel-2 L2A avec filtrage nuageux.

## `run_downloads.py`

Lance automatiquement tout le pipeline :

```bash
python run_downloads.py
```

---

# Organisation des données téléchargées

```text id="acq01"
data/
└── raw/
    ├── Sentinel-1/
    └── Sentinel-2/
```
---
# Configuration EODAG

Le téléchargement nécessite un fichier local de configuration :

`configs/eodag.yml`

Un modèle est fourni :

`configs/eodag.template.yml`

---

# Outils utilisés

* Python
* EODAG
* GeoPandas
* Shapely
* Copernicus Data Space

---

# Validation

Des tests ont été réalisés sur :

* 2 zones côtières
* 20 jours d’acquisition

Les scripts ont montré un fonctionnement robuste face aux erreurs réseau ponctuelles.
