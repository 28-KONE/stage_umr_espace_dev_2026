# Training

This module contains the scripts used to train and evaluate the deep-learning models developed for mangrove mapping.

The training workflow is organized around two main tasks:

1. **Mangrove segmentation**, which aims to distinguish mangrove areas from other land-cover classes.
2. **Habitat segmentation**, which aims to classify the different mangrove habitats.

The directory is organized as follows:

```text
training/
├── mangrove_segmentation/
│   ├── experiments/
│   └── selected_models/
│
└── habitat_segmentation/
```

## 1. Mangrove segmentation

The `mangrove_segmentation/` directory contains the scripts developed for the segmentation of mangrove areas.

Two subdirectories are used to distinguish the different stages of model experimentation.

### Experiments

The `experiments/` directory contains the different architectures and training configurations evaluated during the project.

The tested approaches include:

- CROMA with a CNN segmentation head
- DeepLabV3
- ResNet50 with UPerNet

Experiments were conducted using training data covering French Guiana, Suriname, and Amapá. Additional experiments include *terra firme* forest samples in order to improve the discrimination between mangroves and inland tropical forests.

These scripts are kept to document the different model configurations investigated during the project.

### Selected models

The `selected_models/` directory contains the model configurations retained after the experimental phase.

The selected approach is based on **CROMA representations combined with a UPerNet segmentation architecture**.

Two training configurations are provided:

- training using data from French Guiana, Suriname, and Amapá;
- training using the same geographical areas with additional *terra firme* forest samples.

These scripts correspond to the configurations selected for the final mangrove segmentation experiments.

## 2. Habitat segmentation

The `habitat_segmentation/` directory contains the scripts developed for the classification of mangrove habitats.

The experiments are based on **CROMA representations combined with UPerNet** and investigate different training strategies:

- standard CROMA-UPerNet training;
- CROMA-UPerNet with an auxiliary branch;
- CROMA-UPerNet with partial fine-tuning.

These experiments aim to evaluate different strategies for distinguishing the mangrove habitat classes available in the reference data.


> **Note:** All experiments and processing steps in this project were performed on the Jean Zay supercomputer. The paths and directory structure used in the scripts are specific to this computing environment and must be adapted before running the code on another system.
