import os
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

BASE_WORK = Path("/lustre/fswork/projects/rech/rsn/uuj39cx")
BASE_SCRATCH = Path(os.environ["SCRATCH"])

CSV = BASE_WORK / "Mangrove_CROMA_Project/data/inventory_s2.csv"

S2_RAW_DIR = BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/S2_raw"

NPY_DIR = BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/composite_bands_seasonal_npy"
NPY_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B01","B02","B03","B04","B05","B06", "B07","B08","B8A","B09","B11","B12"]


# SAISONS
SEASONS = {
    "JFM": [1,2,3],
    "AMJ": [4,5,6],
    "JAS": [7,8,9],
    "OND": [10,11,12]}


# INVENTORY
df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="raise")
df["month"] = df["date"].dt.month
print(df["month"].value_counts().sort_index())

tiles = sorted(df.tile.unique())
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

if task_id >= len(tiles):
    raise IndexError(f"Task {task_id} hors range")

tile = tiles[task_id]

START_BAND = int(os.environ["START_BAND"])
END_BAND = int(os.environ["END_BAND"])

print("\n========================")
print(f"TILE : {tile}")
print("========================")
print(f"Bandes : {START_BAND} -> {END_BAND}")

block_size = 512
MAX_STACK = 10

for season_name, months in SEASONS.items():
    print("\n===================================")
    print(f"SEASON : {season_name}")
    print("===================================")

    tile_df = df[(df.tile == tile) & (df.cloud_ratio < 0.95) & (df.month.isin(months))].copy()
    print(f"{len(tile_df)} acquisitions retenues")

    if len(tile_df) == 0:
        print("⚠ Aucun produit pour cette saison")
        continue

    tif_paths = [Path(p)  for p in tile_df.safe_path]
    print(f"{len(tif_paths)} TIFF valides")

    if len(tif_paths) == 0:
        continue


    with rasterio.open(tif_paths[0]) as ref:
        H = ref.height
        W = ref.width

    # COMPOSITE PAR BANDE
    for band_idx in range(START_BAND, END_BAND + 1):
        band_name = S2_BANDS[band_idx - 1]
        npy_file = NPY_DIR / f"{tile}_{season_name}_{band_name}.npy"

        if npy_file.exists():
            print(f"SKIP {season_name} {band_name}")
            continue

        print(f"\nBand {band_name}")
        composite = np.zeros((H, W), dtype=np.float32)

        for row in range(0, H, block_size):
            h = min(block_size, H-row)

            for col in range(0, W, block_size):
                w = min(block_size, W-col)
                window = rasterio.windows.Window(col, row, w, h)
                stack_block = []

                for tif_path in tif_paths:
                    with rasterio.open(tif_path) as src:
                        arr_block = src.read(band_idx, window=window).astype(np.float32)

                    stack_block.append(arr_block)
                    if len(stack_block) >= MAX_STACK:
                        stack_block = [np.nanmedian(np.stack(stack_block), axis=0)]

                if len(stack_block) == 0:
                    continue

                if len(stack_block) > 1:
                    stack_block = [np.nanmedian(np.stack(stack_block), axis=0)]

                med_block = stack_block[0]
                med_block = np.nan_to_num(med_block, nan=0)
                composite[row:row+h, col:col+w] = med_block


        np.save(npy_file, composite.astype(np.float32))
        print(f"Saved {npy_file}")

print(f"\nTile {tile} finished")
