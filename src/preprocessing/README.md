# Preprocessing

This module contains the scripts used to preprocess Sentinel-1 and Sentinel-2 imagery and to construct the temporal composites used as inputs to the deep-learning models.

The preprocessing workflow is divided into three main steps.

## 1. Image preprocessing

The first step prepares the downloaded Sentinel-1 and Sentinel-2 products before temporal compositing. The corresponding scripts are:

- `preprocessing_s1.py`
- `preprocessing_s2.py`

These scripts prepare the satellite observations required by the subsequent compositing step.

## 2. Temporal composite generation

Once the individual satellite observations have been preprocessed, temporal composites are generated using:

- `build_s1_seasonal_composite.py`
- `build_s2_seasonal_composite.py`

The composites are constructed **band by band rather than processing all bands simultaneously**.

This implementation was adopted for computational efficiency. Processing each band independently reduces the amount of data that must be kept in memory at the same time and makes it possible to parallelize the computation across bands.

Each processed band is temporarily saved as a NumPy (`.npy`) file. This intermediate representation also avoids keeping the complete multiband composite in RAM during the compositing process.

The workflow can therefore be summarized as:

```text
Band 1 ──► composite ──► band_1.npy
Band 2 ──► composite ──► band_2.npy
Band 3 ──► composite ──► band_3.npy
  ...           ...
Band N ──► composite ──► band_N.npy
```

## 3. Band aggregation

After all bands have been processed independently, the intermediate `.npy` files are assembled into the final multiband raster products.

This step is performed using:

- `bands_s1_gatherall.py`
- `bands_s2_gatherall.py`

These scripts gather the independently generated bands and write them into multiband **GeoTIFF** files that can subsequently be used for dataset construction and model training.

The complete workflow is therefore:

```text
Sentinel-1 / Sentinel-2
          │
          ▼
   Preprocessing
          │
          ▼
Band-wise temporal composites
          │
          ▼
     .npy files
          │
          ▼
    Band aggregation
          │
          ▼
Multiband GeoTIFF composites
```

This band-wise strategy was designed to make the preprocessing pipeline more suitable for large satellite datasets by reducing memory requirements, facilitating parallel computation, and avoiding the need to manipulate all spectral or radar bands simultaneously.
