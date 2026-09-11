import os
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd

from shapely.geometry import box

# PATHS
BASE_SCRATCH = Path(os.environ["SCRATCH"])
CSV = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/inventory_s1_suriname.csv")
NPY_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/S1_suriname_seasonal")
OUT_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/S1_suriname_composites")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SEASONS
SEASON_MAP = {
"Q1": "JFM",
"Q2": "AMJ",
"Q3": "JAS",
"Q4": "OND"}

# ZONES
task_id = int(os.environ.get( "SLURM_ARRAY_TASK_ID", 0))
zones = [
    "suriname_zone_1",
    "suriname_zone_2",
    "suriname_zone_3",
    "suriname_zone_4",
    "suriname_zone_5",
    "suriname_zone_6"]
zone = zones[task_id]
print(f"\nZONE = {zone}")

# RECONSTRUCTION PROFILE
df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="raise")

zones_gdf = gpd.GeoDataFrame({"zone": zones},
    geometry=[
        box(-58.666992, 5.134715, -57.831031, 7.057282),
        box(-57.831031, 5.134715, -56.995070, 7.057282),
        box(-56.995070, 5.134715, -56.159109, 7.057282),
        box(-56.159109, 5.134715, -55.323148, 7.057282),
        box(-55.323148, 5.134715, -54.487187, 7.057282),
        box(-54.487187, 5.134715, -53.657227, 7.057282),
    ],crs="EPSG:4326")

# TROUVER UNE IMAGE DE REFERENCE
ref_file = None
for _, row in df.iterrows():
    try:
        g = gpd.GeoDataFrame(geometry=[box(row.xmin, row.ymin, row.xmax, row.ymax)], crs=row.crs)
        g = g.to_crs("EPSG:4326")
        geom = g.geometry.iloc[0]
        best = None
        best_area = 0

        for _, z in zones_gdf.iterrows():
            area = geom.intersection(z.geometry).area
            if area > best_area:
                best_area = area
                best = z.zone

        if best == zone:
            ref_file = row.path
            break

    except Exception:
        continue

ref_file = df.iloc[0]["path"]
print(f"Reference : {ref_file}")
with rasterio.open(ref_file) as src:
    profile = src.profile.copy()

profile.update(count=2, dtype="float32", compress="LZW", BIGTIFF="YES")

# SEASONS
for season, season_name in SEASON_MAP.items():
    print("")
    print("=" * 60)
    print(f"{zone} - {season}")
    print("=" * 60)

    vv_file = (NPY_DIR / f"{zone}_{season}_VV.npy")
    vh_file = (NPY_DIR /f"{zone}_{season}_VH.npy")

    if not vv_file.exists():
        print(f"Missing {vv_file}")
        continue

    if not vh_file.exists():
        print(f"Missing {vh_file}")
        continue

    print("Loading VV...")
    vv = np.load(vv_file, mmap_mode="r")

    print("Loading VH...")
    vh = np.load(vh_file, mmap_mode="r")

    assert vv.shape == vh.shape

    print("VV shape :", vv.shape)
    print("VH shape :", vh.shape)

    profile_local = profile.copy()
    profile_local.update(count=2,dtype="float32",height=vv.shape[0],width=vv.shape[1],compress="LZW",BIGTIFF="YES")

    # GEOTIFF
    tif_file = (OUT_DIR / f"{zone}_{season_name}_S1_median.tif")

    BLOCK = 1024

    with rasterio.open(tif_file, "w", **profile_local) as dst:
        for row in range(0, vv.shape[0], BLOCK):
            h = min(BLOCK, vv.shape[0] - row)
            window = rasterio.windows.Window(0, row, vv.shape[1], h)
            dst.write(vv[row:row+h, :], 1, window=window)
            dst.write(vh[row:row+h, :], 2, window=window)

    print(f"Saved {tif_file}")

    del vv
    del vh

print("\nDONE")
