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
```

at 10 m spatial resolution.

---
## 5. Temporal composites

To reduce the impact of cloud contamination, Sentinel-1 and Sentinel-2 acquisitions are aggregated into temporal composites. The compositing strategy is designed to provide spatially complete and temporally consistent observations over the study area.

For each temporal period:

- Sentinel-1 acquisitions are grouped and combined;
- Sentinel-2 acquisitions are filtered according to cloud information;
- available observations are spatially aligned;
- representative composite images are generated;
- Sentinel-1 and Sentinel-2 composites are matched before model training.

The resulting products provide multimodal observations at 10 m spatial resolution and are used as inputs for the segmentation models.

Temporal compositing is particularly important over French Guiana where persistent cloud cover can strongly reduce the availability of usable Sentinel-2 observations.

---

## 6. Binary mangrove segmentation

The first modelling task focuses on the automatic delineation of mangrove extent.

This task is formulated as a **binary semantic segmentation problem**:

```text
0 → non-mangrove
1 → mangrove
```

The objective is to predict a dense mangrove mask from multimodal Sentinel-1 and Sentinel-2 imagery.

### Dataset construction

The binary segmentation dataset is generated by spatially intersecting Sentinel composites with a reference mangrove map.

Image patches are extracted at:

```text
128 × 128 pixels
```

with a spatial resolution of:

```text
10 m
```

Each patch therefore covers approximately:

```text
1.28 km × 1.28 km
```

Patches contain both Sentinel-1 and Sentinel-2 information and are associated with a binary reference mask.

A minimum proportion of mangrove pixels can be used during patch selection to avoid generating a dataset dominated by patches containing little or no mangrove information.

### Multimodal inputs

The model inputs combine:

```text
Sentinel-1
├── VV
└── VH

Sentinel-2
├── multispectral bands
└── resampled to a common 10 m grid
```

Sentinel-1 and Sentinel-2 observations are temporally paired before patch extraction.

Only acquisitions sufficiently close in time are associated in order to limit temporal inconsistencies between both modalities.

### CROMA representations

The binary segmentation experiments investigate the use of **CROMA**, a pretrained multimodal foundation model for Earth observation.

CROMA jointly encodes:

- Sentinel-1 SAR information;
- Sentinel-2 multispectral information.

The model provides several representations derived from:

- SAR tokens;
- optical tokens;
- joint multimodal tokens.

These representations are converted back into spatial feature maps and used by downstream segmentation architectures.

### Segmentation architectures

Several decoder architectures are investigated on top of the extracted representations.

Experiments include convolutional and semantic-segmentation architectures such as:

- UPerNet-based decoding;
- DeepLabV3+;
- ResNet-based feature extraction;
- pretrained Earth observation representations.

The objective is to compare different strategies for transforming Sentinel-1 / Sentinel-2 representations into dense mangrove probability maps.

### Training strategy

The binary segmentation models are trained using image patches and their associated reference masks.

The general workflow is:

```text
Sentinel-1 ─┐
            ├── multimodal representation
Sentinel-2 ─┘
                     │
                     ▼
              feature extraction
                     │
                     ▼
             segmentation decoder
                     │
                     ▼
             mangrove probability
                     │
                     ▼
              binary prediction
```

Training and validation subsets are spatially separated in order to reduce geographical leakage between model development and evaluation.

### Evaluation

Binary segmentation performance is assessed using pixel-wise metrics including:

- Precision;
- Recall;
- F1-score;
- Intersection over Union (IoU).

---

## 7. Habitat classification approaches

After mangrove extent mapping, a second modelling stage focuses on distinguishing ecological habitat classes within mangrove areas.

### CROMA

A major part of the project investigates **CROMA**, a pretrained multimodal foundation model designed for Earth observation data.

CROMA jointly processes:

- Sentinel-1 SAR imagery;
- Sentinel-2 optical imagery.

The pretrained encoder is used to generate multimodal representations that are subsequently exploited for semantic segmentation.

For the main habitat-classification experiments, the pretrained CROMA encoder is kept frozen and connected to a trainable segmentation decoder.

### UPerNet-based decoder

CROMA representations are projected and processed using a decoder inspired by the UPerNet architecture.

The segmentation pipeline includes:

```text
Sentinel-1 ─┐
            ├── CROMA encoder
Sentinel-2 ─┘
                  │
                  ▼
         multimodal features
                  │
                  ▼
            projection layer
                  │
                  ▼
       Pyramid Pooling Module
                  │
                  ▼
          segmentation head
                  │
                  ▼
        pixel-wise prediction
```

### Auxiliary spectral indices

Additional experiments investigate whether explicitly derived vegetation indices can complement the representations learned by CROMA.

The currently evaluated indices are:

- MTCI — MERIS Terrestrial Chlorophyll Index;
- NDMI — Normalized Difference Moisture Index;
- red-edge NDVI;
- NDVI — Normalized Difference Vegetation Index.

These indices are processed through a dedicated convolutional branch and fused with the features produced by the main segmentation network.

## 12. Results

### 12.1 Binary mangrove segmentation

The first set of experiments evaluates the detection of mangrove and non-mangrove areas from Sentinel-1 and Sentinel-2 imagery.

Four segmentation configurations were evaluated:

- **CROMA + CNN**;
- **CROMA + UPerNet**;
- **DeepLabV3+**;
- **ResNet-50 + UPerNet**.

Different probability thresholds were evaluated between 0.50 and 0.65 to analyze the trade-off between precision and recall.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|---:|---:|
| **CROMA + CNN** | 0.50 | 0.8655 | 0.7508 | **0.9336** | 0.8323 | 0.7127 |
| | 0.55 | 0.8742 | 0.7705 | 0.9231 | 0.8399 | 0.7240 |
| | 0.60 | 0.8777 | 0.7988 | 0.8795 | 0.8372 | 0.7200 |
| | 0.65 | **0.8855** | **0.8241** | 0.8644 | **0.8437** | **0.7297** |
| **CROMA + UPerNet** | 0.50 | 0.9431 | 0.9061 | **0.9380** | **0.9218** | **0.8549** |
| | 0.55 | **0.9433** | 0.9125 | 0.9307 | 0.9215 | 0.8545 |
| | 0.60 | 0.9431 | 0.9190 | 0.9222 | 0.9206 | 0.8529 |
| | 0.65 | 0.9422 | **0.9254** | 0.9118 | 0.9186 | 0.8494 |
| **DeepLabV3+** | 0.50 | 0.9219 | 0.8484 | **0.9517** | 0.8971 | 0.8134 |
| | 0.55 | 0.9250 | 0.8615 | 0.9415 | 0.8997 | 0.8178 |
| | 0.60 | 0.9269 | 0.8734 | 0.9303 | **0.9009** | **0.8198** |
| | 0.65 | **0.9277** | **0.8853** | 0.9163 | 0.9006 | 0.8191 |
| **ResNet-50 + UPerNet** | 0.50 | 0.9272 | 0.8544 | **0.9600** | 0.9041 | 0.8251 |
| | 0.55 | 0.9299 | 0.8645 | 0.9534 | 0.9068 | 0.8295 |
| | 0.60 | 0.9322 | 0.8746 | 0.9459 | 0.9089 | 0.8330 |
| | 0.65 | **0.9338** | **0.8847** | 0.9369 | **0.9101** | **0.8350** |

Among the evaluated configurations, **CROMA + UPerNet provides the strongest overall binary segmentation performance**.

At a threshold of 0.50, it reaches:

- **F1-score: 0.9218**;
- **IoU: 0.8549**;
- **Recall: 0.9380**;
- **Precision: 0.9061**.

The threshold analysis also shows the expected precision-recall trade-off. Increasing the threshold generally increases precision while reducing recall.

The alternative architectures remain competitive. **ResNet-50 + UPerNet** reaches an IoU of **0.8350** at threshold 0.65, while **DeepLabV3+** reaches **0.8198** at threshold 0.60. The simpler **CROMA + CNN** configuration performs substantially below the decoder-based architectures, with a maximum IoU of **0.7297**.

#### Terra-firme forest hard negatives

An additional experiment was conducted to investigate one of the main sources of confusion in the binary segmentation task: the spectral and structural similarity between mangroves and inland terra-firme forests.

Terra-firme forest pixels were introduced as explicit hard-negative examples during training while preserving the binary formulation of the problem: **mangrove vs. non-mangrove**.

At a probability threshold of 0.50, the resulting CROMA + UPerNet model was compared with the original configuration:

| Configuration | Accuracy | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|---:|
| **CROMA + UPerNet** | 0.9431 | **0.9061** | 0.9380 | 0.9218 | 0.8549 |
| **CROMA + UPerNet + terra-firme forest hard negatives** | **0.9488** | 0.9044 | **0.9579** | **0.9304** | **0.8698** |

Introducing terra-firme forest hard negatives improves the overall binary segmentation performance. The IoU increases from **0.8549 to 0.8698**, while the F1-score increases from **0.9218 to 0.9304**. The largest improvement is observed for recall, which increases from **0.9380 to 0.9579**, with only a small decrease in precision from **0.9061 to 0.9044**.

The model therefore detects a larger proportion of mangrove pixels while maintaining a similar level of precision.

However, confusion with terra-firme forests remains substantial. Among the **151,220 terra-firme forest pixels** evaluated, **94,335 were predicted as mangrove**, corresponding to a forest-to-mangrove false-positive rate of **62.38%**.

This result indicates that introducing terra-firme forests as hard negatives improves the global segmentation metrics but does not fully resolve the specific confusion between mangrove and inland forest.

#### Perspectives

A natural extension would be to reformulate the binary segmentation problem as a multiclass task in which terra-firme forest is represented as an explicit semantic class, for example:

**mangrove / terra-firme forest / other non-mangrove**.

Such a formulation could allow the model to learn a dedicated representation of terra-firme forests instead of grouping them with all other non-mangrove land-cover types. This could potentially improve the discrimination between mangrove and inland forest, although this hypothesis would need to be evaluated experimentally.

---

### 12.2 Multiclass mangrove habitat segmentation

The second set of experiments focuses on the classification of mangrove areas into four ecological stages:

```text
Young → Adult → Mature → Senescent
```

Two main architectures were evaluated using the same grouped spatial train, validation and test partition:

- **CROMA + UPerNet**;
- **ResNet-50 + DeepLabV3+**.

#### CROMA + UPerNet

The CROMA-based model achieved a best validation mIoU of:

```text
0.3919
```

On the independent test set, the following class-wise performances were obtained:

| Habitat class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Young | 0.5716 | 0.4587 | 0.5090 | 0.3414 |
| Adult | 0.4439 | 0.2823 | 0.3451 | 0.2086 |
| Mature | 0.6135 | 0.7532 | 0.6762 | 0.5108 |
| Senescent | 0.4838 | 0.5100 | 0.4966 | 0.3303 |

Overall test performance:

| Metric | Value |
|---|---:|
| Macro F1-score | **0.5067** |
| Mean IoU | **0.3478** |

The **Mature** class is the best recognized habitat stage, reaching an IoU of **0.5108** and an F1-score of **0.6762**.

The **Adult** class remains the most difficult to distinguish, with an IoU of **0.2086**. Confusion analysis shows that a substantial proportion of Adult pixels are assigned to the Mature class.

These results illustrate the difficulty of distinguishing ecological stages that form a gradual vegetation succession rather than strictly separated land-cover categories.

#### ResNet-50 + DeepLabV3+

A ResNet-50 + DeepLabV3+ architecture was evaluated as an alternative baseline.

| Habitat class | F1-score | IoU |
|---|---:|---:|
| Young | 0.3992 | 0.2494 |
| Adult | 0.2068 | 0.1153 |
| Mature | 0.6183 | 0.4475 |
| Senescent | 0.4419 | 0.2836 |

Overall test performance:

| Metric | Value |
|---|---:|
| Macro F1-score | **0.4165** |
| Mean IoU | **0.2739** |

Under the experimental protocol used in this study, **CROMA + UPerNet outperforms the ResNet-50 + DeepLabV3+ baseline across the three global evaluation metrics**.

Compared with ResNet-50 + DeepLabV3+, CROMA + UPerNet increases:

```text
Macro F1 : 0.4165 → 0.5067
mIoU     : 0.2739 → 0.3478
```


This suggests that the multimodal representations extracted from Sentinel-1 and Sentinel-2 by CROMA provide more informative features for distinguishing mangrove habitat stages under the evaluated configuration.

#### Auxiliary spectral-index experiment

A complementary experiment investigates whether explicit vegetation information can further improve habitat classification.

Four vegetation and moisture indices were introduced through an auxiliary convolutional branch:

- **MTCI**;
- **NDMI**;
- **red-edge NDVI (NDVI_RE)**;
- **NDVI**.

The auxiliary features were fused with the multimodal representations extracted by CROMA before the final segmentation head. The original ordinal cross-entropy loss and the same spatial train, validation and test partition were retained in order to isolate the contribution of the additional spectral information.

On the independent test set, the following performances were obtained:

| Habitat class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Young | 0.7385 | 0.3702 | 0.4932 | 0.3273 |
| Adult | 0.4776 | 0.2327 | 0.3130 | 0.1855 |
| Mature | 0.5838 | 0.8505 | 0.6924 | 0.5295 |
| Senescent | 0.5393 | 0.5262 | 0.5326 | 0.3630 |

Overall test performance:

| Metric | CROMA + UPerNet | + Auxiliary indices |
|---|---:|---:|
| Macro F1-score | 0.5067 | **0.5078** |
| Mean IoU | 0.3478 | **0.3513** |

The auxiliary indices lead to only a marginal improvement in the global segmentation metrics. Macro F1 increases from **0.5067 to 0.5078** and mean IoU from **0.3478 to 0.3513**.

However, this improvement is not uniform across habitat stages. The Mature and Senescent classes improve, while Young and Adult perform worse than with the original CROMA + UPerNet configuration:

| Habitat class | F1 original | F1 + indices | IoU original | IoU + indices |
|---|---:|---:|---:|---:|
| Young | **0.5090** | 0.4932 | **0.3414** | 0.3273 |
| Adult | **0.3451** | 0.3130 | **0.2086** | 0.1855 |
| Mature | 0.6762 | **0.6924** | 0.5108 | **0.5295** |
| Senescent | 0.4966 | **0.5326** | 0.3303 | **0.3630** |

The confusion analysis also reveals an increased tendency to assign pixels to the Mature class. In particular, the proportion of Adult pixels classified as Mature increases from **41.18% to 53.81%**, while Young-to-Mature confusion increases from **36.04% to 44.71%**.

These results suggest that the explicit spectral indices provide complementary information, particularly for ordinal consistency and the later habitat stages, but they do not resolve the main confusion between ecological stages. Instead, they reinforce the tendency of the model to favor the Mature class.

#### Partial fine-tuning and focal-loss experiment

A final experiment investigated whether adapting part of the pretrained CROMA encoder to the habitat segmentation task could improve the discrimination between ecological stages.

Instead of keeping the complete CROMA encoder frozen, the last transformer blocks of the Sentinel-1, Sentinel-2 and joint SAR-optical branches were fine-tuned using a lower learning rate than
the segmentation decoder.

A focal classification loss combined with the ordinal component was also evaluated in this configuration.

The resulting model reached a best validation mIoU of:

```text
0.3929
```

On the independent test set, the following performances were obtained:

| Habitat class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Young | 0.6616 | 0.3874 | 0.4887 | 0.3234 |
| Adult | 0.4229 | 0.2773 | 0.3350 | 0.2012 |
| Mature | 0.5972 | 0.8042 | 0.6854 | 0.5214 |
| Senescent | 0.4960 | 0.4803 | 0.4880 | 0.3228 |

Overall test performance:

| Metric | CROMA + UPerNet | Partial fine-tuning + focal loss |
|---|---:|---:|
| Macro F1-score | **0.5067** | 0.4993 |
| Mean IoU | **0.3478** | 0.3422 |

Partial fine-tuning combined with focal loss does not improve the overall segmentation performance compared with the original frozen CROMA configuration. Macro F1 decreases from **0.5067 to 0.4993** and mean IoU from **0.3478 to 0.3422**.

The Mature class again benefits the most from the modified training strategy, reaching an IoU of **0.5214**, compared with **0.5108** for the original model. However, the other three habitat classes obtain lower IoU values.

The confusion matrix confirms that the tendency to predict the Mature class remains. Adult-to-Mature confusion increases from **41.18% to 49.27%**, Young-to-Mature confusion from **36.04% to 40.40%**, and Senescent-to-Mature confusion from **30.44% to 33.22%**.

Because both the encoder fine-tuning strategy and the classification loss were modified in this experiment, their individual contributions cannot be isolated from these results. The experiment should therefore be interpreted as an evaluation of the combined configuration rather than as evidence that partial fine-tuning alone is detrimental.

#### Overall comparison

The main habitat-segmentation experiments can be summarized as follows:

| Configuration | Macro F1 | Mean IoU | 
|---|---:|---:|---:|
| ResNet-50 + DeepLabV3+ | 0.4165 | 0.2739 |
| CROMA + UPerNet | 0.5067 | 0.3478 | 
| **CROMA + UPerNet + auxiliary indices** | **0.5078** | **0.3513** | 
| CROMA partial fine-tuning + UPerNet + focal loss | 0.4993 | 0.3422 | 

Overall, **CROMA + UPerNet with auxiliary spectral indices achieves the best global results**, although the improvements in Macro F1 and mean IoU over the original CROMA + UPerNet model remain small.

More generally, the experiments consistently show that the **Mature** class is the easiest habitat stage to identify, whereas **Adult** is the most difficult. Several model variants increase the recognition of Mature pixels but simultaneously increase the tendency to assign pixels from the other habitat stages to this class.

This persistent confusion suggests that the remaining difficulty may not be explained solely by model architecture or loss formulation. Mangrove habitat stages represent a gradual ecological succession, with transitional areas and potentially mixed pixels at Sentinel spatial resolution. Consequently, the spectral and radar signatures of neighboring stages may overlap substantially.

#### Perspectives

Further work could investigate the intrinsic separability of the four habitat stages from Sentinel-1 and Sentinel-2 observations before introducing additional model complexity.

Possible directions include analyzing class distributions and confusion patterns across geographical areas, evaluating the effect of spatial resolution and mixed pixels, and investigating whether
alternative habitat groupings provide a more robust representation of the ecological gradients.

Higher-resolution observations, when available, could also help assess whether some of the remaining confusion originates from the 10 m spatial resolution of the Sentinel data rather than from the
segmentation architecture itself.

---

## 13. Repository structure

The repository is organized according to the main stages of the processing and modelling workflow:

```text
stage_umr_espace_dev_2026/
│
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml
│
├── configs/
│   └── ...
│
├── src/
│   │
│   ├── data_acquisition/
│   │   ├── README.md
│   │   ├── generate_bboxes_mangrove.py
│   │   ├── download_sentinel1.py
│   │   ├── download_sentinel2.py
│   │   └── run_downloads.py
│   │
│   ├── preprocessing/
│   │   └── ...
│   │
│   ├── temporal_composites/
│   │   └── ...
│   │
│   ├── datasets/
│   │   └── ...
│   │
│   ├── models/
│   │   ├── binary_segmentation/
│   │   └── habitat_classification/
│   │
│   ├── training/
│   │   └── ...
│   │
│   ├── evaluation/
│   │   └── ...
│   │
│   ├── inference/
│   │   └── ...
│   │
│   └── utils/
│       └── ...
│
├── notebooks/
│   └── ...
│
├── docs/
│   └── ...
│
├── tests/
│   └── ...
│
└── outputs/
    └── ...
```

The repository intentionally separates data acquisition, preprocessing, temporal compositing, dataset construction, model development, training, evaluation and inference to facilitate reproducibility and reuse.
