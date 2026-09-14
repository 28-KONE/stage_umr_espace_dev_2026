# Data acquisition

This module contains the scripts developed to automatically download the data used in this project.

Data are acquired from:

- **Copernicus Data Space Ecosystem (CDSE)**;
- **Microsoft Planetary Computer**.

The scripts automate the search and download of:

- **Sentinel-1 GRD** products, including VV and VH polarizations;
- **Sentinel-2 Level-2A** products, including multispectral bands and the Scene Classification Layer (SCL).

Data are downloaded for the different geographical areas defined in the study and for the temporal periods required to construct the dataset.

> **Note:** All experiments and processing steps in this project were performed on the Jean Zay supercomputer. The paths and directory structure used in the scripts are specific to this computing environment and must be adapted before running the code on another system.
