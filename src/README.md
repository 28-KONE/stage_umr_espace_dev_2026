# Source code

This directory contains the main source code developed during the project.

The processing pipeline is organized into three main modules:

- `data_acquisition/`: scripts used to download and organize Sentinel-1 and Sentinel-2 satellite imagery.
- `preprocessing/`: scripts used to preprocess the satellite data and generate the temporal composites required for the deep-learning models.
- `training/`: scripts used to train and evaluate the models for mangrove segmentation and mangrove habitat classification.

Each module contains its own `README.md` providing additional information about the corresponding scripts and workflow.

## Workflow overview

The general organization of the source code follows the main stages of the project:

```text
Sentinel-1 / Sentinel-2
          │
          ▼
   Data acquisition
          │
          ▼
     Preprocessing
          │
          ▼
  Temporal composites
          │
          ▼
     Model training
          │
     ┌────┴────┐
     ▼         ▼
  Mangrove   Mangrove habitat
segmentation  classification
```text

