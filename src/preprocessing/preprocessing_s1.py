from pathlib import Path
import rasterio
import numpy as np
from scipy.ndimage import uniform_filter
import os
import torch


# PATHS
WORK = Path(os.environ["WORK"])
INPUT_DIR = WORK / "Mangrove_CROMA_Project/data/processed/Sentinel-1_geo"
OUTPUT_DIR = WORK / "Mangrove_CROMA_Project/data/processed/Sentinel-1_pc_georef"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FUNCTIONS
def calibrate_sigma0(img):
    img = img.astype(np.float32)
    img = np.where(img <= 0, np.nan, img)
    return img


def to_db(img):
    return 10 * np.log10(img)


def speckle_filter(img, size=3):
    return uniform_filter(img, size=size)


# LOAD SCENES
scenes = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
print(f"\n Found {len(scenes)} scenes\n")


# SLURM TASK ID
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
if task_id >= len(scenes):
    print(f"Task {task_id} out of range")
    exit()

scene_dir = scenes[task_id]
i = task_id

try:
    vv_path = scene_dir / "VV.tif"
    vh_path = scene_dir / "VH.tif"

    if not vv_path.exists() or not vh_path.exists():
        print(f"⚠ Missing VV or VH in {scene_dir.name}")
        exit()

    out_path = OUTPUT_DIR / f"{scene_dir.name}.tif"
    if out_path.exists():
        print(f"[{i}]  Already processed: {scene_dir.name}")
        exit()

    print(f"[{i}] Processing: {scene_dir.name}")

    # READ + PROCESS PAR BLOCS
    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:

        if src_vv.crs is None or src_vv.transform is None:
            print(f"❌ Missing georef in RAW: {scene_dir.name}")
            exit()

        profile = src_vv.profile.copy()
        profile.update(dtype=rasterio.float32, count=2, crs=src_vv.crs, transform=src_vv.transform)

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in src_vv.block_windows(1):
                vv = src_vv.read(1, window=window).astype(np.float32)
                vh = src_vh.read(1, window=window).astype(np.float32)

                # STACK
                img = np.stack([vv, vh])

                # CALIBRATION
                img = calibrate_sigma0(img)

                # dB
                img_db = to_db(img)

                # SPECKLE
                img_filtered = np.stack([speckle_filter(band, size=3) for band in img_db])

                img_filtered = np.nan_to_num(img_filtered, nan=0)

                # ÉCRITURE
                dst.write(img_filtered.astype(np.float32), window=window)
    meta = []
    meta.append({
        "scene": scene_dir.name,
        "transform": src_vv.transform,
        "crs": str(src_vv.crs),
        "width": src_vv.width,
        "height": src_vv.height})

    np.savez(OUTPUT_DIR / f"metadata_{scene_dir.name}.npz", **meta)

    print(" Done")

except Exception as e:
    print(f"❌ Error on {scene_dir.name}: {e}")

print("\n✔ Task finished!\n")
