from pathlib import Path
import os
import logging

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ENV
BASE_SCRATCH = Path(os.environ["SCRATCH"])

LOG_DIR = BASE_SCRATCH / "Mangrove_CROMA_Project/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "preprocess_s2.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])



# PATHS
RAW_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/raw/Sentinel-2")
OUT_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/Sentinel-2_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# BANDS
S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


# SCL CLASSES TO MASK
CLOUD_CLASSES = {
    3,   # cloud shadow
    8,   # medium cloud
    9,   # high cloud
    10,  # cirrus
}


# FUNCTION
def read_and_resample(band_path, ref_profile):
    with rasterio.open(band_path) as src:
        out = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear)

    return out


# MAIN
def main():

    scenes = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logging.info(f"Found {len(scenes)} scenes")

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id >= len(scenes):
        raise IndexError(f"Task {task_id} out of range ({len(scenes)})")

    scene = scenes[task_id]
    logging.info(f"Processing {scene.name}")

    out_file = OUT_DIR / f"{scene.name}.tif"
    if out_file.exists():
        logging.info("Already processed")
        return

    
    # REFERENCE BAND (10 m grid)
    ref_band = scene / "B02.tif"

    if not ref_band.exists():
        raise FileNotFoundError(ref_band)

    with rasterio.open(ref_band) as ref:
        ref_profile = ref.profile.copy()

        ref_profile.update(
            driver="GTiff",
            count=len(S2_BANDS),
            dtype="float32",
            compress="LZW",
            tiled=True,
            BIGTIFF="YES",
            nodata=np.nan)

    
    # CLOUD MASK FROM SCL
    scl_file = scene / "SCL.tif"

    if not scl_file.exists():
        raise FileNotFoundError(scl_file)

    with rasterio.open(scl_file) as src:
        scl_resampled = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint8)
        reproject(
        source=rasterio.band(src, 1),
        destination=scl_resampled,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.nearest)

    cloud_mask = np.isin(scl_resampled, list(CLOUD_CLASSES))
    cloud_ratio = cloud_mask.mean()

    logging.info(f"Cloud ratio = {cloud_ratio:.3f}")

    if cloud_ratio > 0.95:
        logging.warning(f"Skipping ultra-cloudy scene: {scene.name}")
        return

    # WRITE OUTPUT
    try:
        with rasterio.open(out_file, "w", **ref_profile) as dst:
            for band_idx, band_name in enumerate(S2_BANDS, start=1):
                band_file = scene / f"{band_name}.tif"

                if not band_file.exists():
                    raise FileNotFoundError(f"Missing band: {band_name}")

                data = read_and_resample(band_file, ref_profile)
                if data.shape != cloud_mask.shape:
                    raise ValueError(f"Shape mismatch for {band_name}")

                data[cloud_mask] = np.nan
                dst.write(data.astype(np.float32), band_idx)

        logging.info(f"Saved: {out_file.name}")

    except Exception as e:
        logging.error(f"Failed on {scene.name}: {e}")

        if out_file.exists():
            os.remove(out_file)

        raise

    # VALIDATION
    with rasterio.open(out_file) as src:
        test = src.read(1)

        if test.size == 0:
            raise RuntimeError(f"Invalid output: {scene.name}")

    logging.info(f"SUCCESS: {scene.name}")


if __name__ == "__main__":
    main()
