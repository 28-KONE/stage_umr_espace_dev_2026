from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from shapely.geometry import box

from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform
from rasterio.crs import CRS


start_time = time.time()

BASE_SCRATCH = Path(os.environ["SCRATCH"])
CSV = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/inventory_s1.csv")
OUT_DIR = (BASE_SCRATCH / "Mangrove_CROMA_Project/data/processed/S1_suriname_seasonal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "Q1": [1, 2, 3],
    "Q2": [4, 5, 6],
    "Q3": [7, 8, 9],
    "Q4": [10, 11, 12]}

TARGET_CRS = CRS.from_epsg(32622)
BLOCK = 256

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
zones = gpd.GeoDataFrame({
        "zone": [
            "suriname_zone_1",
            "suriname_zone_2",
            "suriname_zone_3",
            "suriname_zone_4",
            "suriname_zone_5",
            "suriname_zone_6",
        ]},
    geometry=[
        box(-58.666992, 5.134715, -57.831031, 7.057282),
        box(-57.831031, 5.134715, -56.995070, 7.057282),
        box(-56.995070, 5.134715, -56.159109, 7.057282),
        box(-56.159109, 5.134715, -55.323148, 7.057282),
        box(-55.323148, 5.134715, -54.487187, 7.057282),
        box(-54.487187, 5.134715, -53.657227, 7.057282),
    ], crs="EPSG:4326")

zone = zones.iloc[task_id].zone

print(f"\nZONE = {zone}")

df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")


# AFFECTATION DES IMAGES AUX ZONES
image_to_zone = {}

for _, row in df.iterrows():
    try:
        g = gpd.GeoDataFrame(geometry=[box(row.xmin, row.ymin, row.xmax, row.ymax)], crs=row.crs)
        g = g.to_crs("EPSG:4326")
        geom = g.geometry.iloc[0]

        for _, z in zones.iterrows():
            if geom.intersects(z.geometry):
                image_to_zone.setdefault(row.path, []).append(z.zone)

    except Exception:
        continue

zone_paths = [Path(p)
    for p, zlist in image_to_zone.items()
    if zone in zlist]

print(f"{zone} : " f"{len(zone_paths)} scenes")

if len(zone_paths) == 0:
    raise RuntimeError(f"Aucune scene pour {zone}")



# REFERENCE UNIQUE POUR LA ZONE
best_file = None
best_pixels = 0

for tif in zone_paths:
    with rasterio.open(tif) as src:
        pixels = (src.width * src.height)

        if pixels > best_pixels:
            best_pixels = pixels
            best_file = tif

print(f"\nREFERENCE = {best_file}")
with rasterio.open(best_file) as ref:
    ref_transform, W, H = (calculate_default_transform(ref.crs, TARGET_CRS, ref.width, ref.height, *ref.bounds))
    count = ref.count

print(f"REF SIZE = {H} x {W}")


datasets = [rasterio.open(tif)  for tif in zone_paths]
path_to_vrt = {}
for src in datasets:
    vrt = WarpedVRT(
        src,
        crs=TARGET_CRS,
        transform=ref_transform,
        width=W,
        height=H,
        resampling=Resampling.nearest)

    path_to_vrt[Path(src.name)] = vrt


# BANDES
band_names = {1: "VV", 2: "VH"}


# SAISONS
for season, months in SEASONS.items():
    print(f"\n===== {season} =====")

    sdf = df[df["date"].dt.month.isin(months)].copy()
    season_paths = [
        Path(p)
        for p, zlist
        in image_to_zone.items()
        if (zone in zlist  and  Path(p)  in zone_paths)]

    season_paths = [p
        for p in season_paths
        if pd.to_datetime(p.name.split("_")[4][:8], format="%Y%m%d").month in months]

    print(f"{len(season_paths)} acquisitions")
    if len(season_paths) == 0:
        continue

    for band_idx in range(1, count + 1):
        band_name = band_names[band_idx]
        npy_file = (OUT_DIR / f"{zone}_{season}_{band_name}.npy")

        if npy_file.exists():
            print(f"SKIP {npy_file.name}")
            continue

        print(f"\nProcessing " f"{season} " f"{band_name}")
        composite = np.full((H, W), np.nan, dtype=np.float32)
        total = (((H + BLOCK - 1) // BLOCK) * ((W + BLOCK - 1) // BLOCK))

        done = 0
        for row0 in range(0, H, BLOCK):
            h = min(BLOCK, H - row0)

            for col0 in range(0, W, BLOCK):
                w = min(BLOCK, W - col0)
                window = Window(col0, row0, w, h)
                stack = []

                for tif in season_paths:
                    vrt = path_to_vrt[tif]
                    arr = vrt.read(band_idx, window=window, out_shape=(h, w)).astype(np.float32)
                    nodata = vrt.nodata

                    if nodata is not None:
                        arr[arr == nodata] = np.nan

                    stack.append(arr)

                if len(stack) == 0:
                    continue

                med = np.nanmedian(np.asarray(stack), axis=0)
                composite[row0:row0+h, col0:col0+w] = med
                done += 1

                if done % 50 == 0:
                    elapsed = (time.time() - start_time) / 60

                    print(
                        f"{zone} "
                        f"{season} "
                        f"{band_name}: "
                        f"{done}/{total} "
                        f"{100*done/total:.1f}% "
                        f"{elapsed:.1f} min")

        np.save(npy_file, composite.astype(np.float32))
        print(f"Saved {npy_file}")

        del composite



for vrt in path_to_vrt.values():
    vrt.close()

for src in datasets:
    src.close()

print("\nDONE")
