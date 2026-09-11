import os
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path


# PATHS
BASE_SCRATCH = Path(os.environ["SCRATCH"])
CSV = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/inventory_s2_suriname.csv")
NPY_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/composite_s2_suriname_seasonal_npy")
OUT_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/S2_suriname_seasonal_composites")
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B01","B02","B03","B04", "B05","B06","B07","B08", "B8A","B09","B11","B12"]
SEASONS = {"JFM": "Jan-Mar", "AMJ": "Apr-Jun", "JAS": "Jul-Sep", "OND": "Oct-Dec"}


# TILE
df = pd.read_csv(CSV)
tiles = sorted(df.tile.unique())
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
tile = tiles[task_id]
print(f"\nTile : {tile}")

# Reference TIFF
tile_df = df[df.tile == tile]
ref_tif = Path(tile_df.iloc[0]["safe_path"])
print("Reference:", ref_tif)

with rasterio.open(ref_tif) as src:
    profile = src.profile.copy()

profile.update(count=12, dtype="float32", compress="LZW", BIGTIFF="YES")


# Rebuild every season
for season_code, season_name in SEASONS.items():

    print("\n===================================")
    print(f"Season : {season_name}")
    print("===================================")

    out_file = OUT_DIR / f"{tile}_{season_code}_composite.tif"
    with rasterio.open(out_file, "w", **profile) as dst:
        for band_idx, band_name in enumerate(S2_BANDS, start=1):
            npy_file = (NPY_DIR / f"{tile}_{season_code}_{band_name}.npy")

            if not npy_file.exists():
                raise FileNotFoundError(f"Missing {npy_file}")

            print(f"Loading {season_code} {band_name}")
            arr = np.load(npy_file)
            dst.write(arr.astype(np.float32), band_idx)

    print(f"Saved -> {out_file}")

print("\nAll seasonal composites generated.")
