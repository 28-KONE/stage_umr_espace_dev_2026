# Deep Learning for Mangrove Mapping from Sentinel-1 and Sentinel-2 Imagery

Repository associated with the 2026 research internship conducted at **UMR Espace-Dev (IRD)** on deep learning methods for mangrove mapping from Earth observation data.

The project investigates the use of **multimodal satellite imagery** and **pretrained representation learning models** for large-scale mangrove monitoring along the Amazon-influenced coast of South America.

The main remote sensing sources are:

- **Sentinel-1** Synthetic Aperture Radar (SAR) imagery;
- **Sentinel-2** multispectral optical imagery.

The work focuses on two main tasks:

1. **Mangrove extent mapping**;
2. **Mangrove habitat classification**.

The repository contains the main scripts used for data acquisition, preprocessing, dataset construction, model training, evaluation and inference.

---

## 1. Scientific context

Mangroves are intertidal forest ecosystems occurring along tropical and subtropical coastlines. They provide important ecosystem services including:

- coastal protection;
- biodiversity support;
- nursery habitats;
- carbon sequestration and storage;
- climate-change mitigation.

Monitoring mangrove extent and ecological condition over large areas remains challenging because field surveys and very-high-resolution imagery are costly and difficult to acquire systematically.

Satellite Earth observation provides a complementary solution. In particular, **Sentinel-1** and **Sentinel-2** offer free, recurrent and spatially extensive observations with complementary information:

- Sentinel-1 provides radar information related to surface structure and moisture and is largely independent of cloud cover;
- Sentinel-2 provides multispectral information related to vegetation properties and spectral characteristics.

The combination of both modalities is therefore particularly relevant in tropical environments where persistent cloud cover can limit optical observations.

---

## 2. Objectives

The general objective of this project is to investigate deep learning approaches for extracting information on mangrove ecosystems from Sentinel-1 and Sentinel-2 imagery at 10 m spatial resolution.

The work is organized around several downstream applications:

### Mangrove extent mapping

Binary semantic segmentation is used to distinguish:

- mangrove;
- non-mangrove.

The objective is to produce spatially consistent mangrove maps from multimodal Sentinel imagery.

### Mangrove habitat classification

Within mapped mangrove areas, a second semantic segmentation task aims to distinguish four ecological stages:

| Class | Habitat stage |
|---|---|
| 0 | Young |
| 1 | Adult |
| 2 | Mature |
| 3 | Senescent |

The classification is performed from Sentinel-1 and Sentinel-2 image patches using deep neural networks.

### Biomass and carbon

The workflow is also designed to provide a basis for subsequent estimation of:

- above-ground biomass;
- carbon stocks.

These components rely on dedicated reference products and constitute an extension of the mapping framework.

---

## 3. Study area

The main study area is the **coast of French Guiana**, characterized by highly dynamic mangrove ecosystems strongly influenced by sediment transport from the Amazon River.

Depending on the experiment, the geographical scope can be extended along the Amazon-influenced coastline, including areas in:

- Suriname;
- northern Brazil.

The spatial extent used for data processing is divided into several geographical zones to facilitate satellite data acquisition, preprocessing and model evaluation.

---

## 4. Satellite data

### Sentinel-1

Sentinel-1 Ground Range Detected (GRD) products are used with the following polarizations:

- VV;
- VH.

The preprocessing workflow includes operations such as:

- orbit correction;
- radiometric calibration;
- terrain correction;
- spatial alignment;
- resampling to 10 m.

### Sentinel-2

Sentinel-2 Level-2A multispectral imagery is used.

The workflow includes the following spectral bands:

| Resolution | Bands |
|---|---|
| 10 m | B02, B03, B04, B08 |
| 20 m | B05, B06, B07, B8A, B11, B12 |
| 60 m | B01, B09 |

Bands are resampled and spatially aligned before being used by the models.

Cloud information is derived from the Sentinel-2 Scene Classification Layer and, depending on the experiment, additional cloud-detection approaches.

---

## 5. Multimodal data preparation

Sentinel-1 and Sentinel-2 acquisitions are temporally paired to construct multimodal observations.

The main processing steps include:

1. satellite product acquisition;
2. Sentinel-1 preprocessing;
3. Sentinel-2 preprocessing;
4. cloud screening;
5. spatial co-registration;
6. temporal pairing;
7. image normalization;
8. reference-map alignment;
9. extraction of fixed-size image patches;
10. construction of train, validation and test datasets.

Image patches currently used for deep learning experiments have a spatial size of:

```text
128 × 128 pixels


