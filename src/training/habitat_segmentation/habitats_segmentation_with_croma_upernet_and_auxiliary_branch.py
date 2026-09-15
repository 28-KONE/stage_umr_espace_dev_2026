import os
import gc
import random
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from shapely.geometry import box
from collections import Counter

from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import shape
from rasterio.transform import from_bounds
from shapely.geometry import Point
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rasterio.warp import reproject, Resampling
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from use_croma import PretrainedCROMA


# SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ENV
os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"

BASE = Path(os.environ["WORK"])
BASE2 = Path(os.environ["SCRATCH"])

# PATHS
S1_DIR = BASE / "Mangrove_CROMA_Project/data/processed/S1_zone_composites_seasonal"
S2_DIR = BASE / "Mangrove_CROMA_Project/data/processed/S2_seasonal_composite"

MANGROVE_VECTOR = BASE / "Mangrove_CROMA_Project/data/ancillary/vecteurs_mangroves_2023/Test_Vect_2023_false.shp"
HABITAT_RASTER  = BASE / "Mangrove_CROMA_Project/data/ancillary/"    "Hab_4_classes/"  "Habitat_mangrove_2023_4classes.tif"

OUT_DIR = BASE / "Mangrove_CROMA_Project/scripts/training/segmentation_habitats/segmentation_dataset_croma_upernet_multiclass_with_auxiliairy_branch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR = OUT_DIR / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_INDEX = OUT_DIR / "total_index.parquet"
TRAIN_INDEX = OUT_DIR / "train_index.parquet"
VAL_INDEX   = OUT_DIR / "val_index.parquet"
TEST_INDEX  = OUT_DIR / "test_index.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "croma_upernet_multiclass.log"),
        logging.StreamHandler()
    ]
)

logging.info(f"Using device: {DEVICE}")

# CONFIG
PATCH_SIZE = 128
STRIDE = 128
BATCH_SIZE = 16
ACCUM_STEPS = 1
EPOCHS = 100
LR = 5e-4
NUM_WORKERS = 2
PIN_MEMORY = True
POS_THRESHOLD = 0.01


# HABITAT DATASET AUDIT
logging.info("")
logging.info("=" * 70)
logging.info("HABITAT DATASET AUDIT")
logging.info("=" * 70)

# CHECK RAW HABITAT RASTER
logging.info("")
logging.info("▶ Inspecting raw habitat raster...")

with rasterio.open(HABITAT_RASTER) as hab:
    logging.info(f"Habitat CRS      : {hab.crs}")
    logging.info(f"Habitat shape    : {hab.height} x {hab.width}")
    logging.info(f"Habitat res      : {hab.res}")
    logging.info(f"Habitat nodata   : {hab.nodata}")
    logging.info(f"Habitat dtype    : {hab.dtypes[0]}")

    habitat_raw = hab.read(1)

unique_values, counts = np.unique(habitat_raw, return_counts=True)

logging.info("")
logging.info("===== RAW HABITAT VALUES =====")

for value, count in zip(unique_values, counts):
    logging.info(f"Value {value}: "  f"{count:,} pixels")


# MAIN
if IS_MAIN_PROCESS:

    # LOAD MANGROVE VECTOR
    logging.info("▶ Loading mangrove polygons...")
    mangrove_gdf = gpd.read_file(MANGROVE_VECTOR)
    mangrove_gdf = mangrove_gdf.to_crs("EPSG:32622")
    mangrove_gdf = mangrove_gdf[mangrove_gdf.geometry.notnull()]
    mangrove_gdf = mangrove_gdf[mangrove_gdf.geometry.is_valid]

    logging.info(f"✔ Mangrove polygons loaded: {len(mangrove_gdf)}")

    # LOAD HABITAT RASTER
    logging.info("▶ Loading habitat raster...")
    habitat_src = rasterio.open(HABITAT_RASTER)

    logging.info(
        f"Habitat raster | "
        f"CRS={habitat_src.crs} | "
        f"RES={habitat_src.res} | "
        f"SIZE={habitat_src.width}x{habitat_src.height} | "
        f"NODATA={habitat_src.nodata}")

    # LOAD EXISTING INDEX OR CREATE IT
    if TOTAL_INDEX.exists():
        df = pd.read_parquet(TOTAL_INDEX)
        logging.info(f"✔ Index loaded: {len(df)} samples")

    else:
        logging.info("▶ Creating multi-class habitat masks + index...")
        rows = []
        s1_files = sorted(S1_DIR.glob("*.tif"))
        s2_files = sorted(S2_DIR.glob("*.tif"))

        logging.info(f"S1 composites found: {len(s1_files)}")
        logging.info(f"S2 composites found: {len(s2_files)}")

        # 4. INDEX S2 COMPOSITES BY SEASON
        logging.info("▶ Indexing S2 composites by season...")

        s2_index = {}
        for s2 in s2_files:
            season = s2.stem.split("_")[-2]

            with rasterio.open(s2) as src:
                geom = box(*src.bounds)
                s2_crs = src.crs

            s2_index.setdefault(season, []).append({"path": s2, "geom": geom, "crs": s2_crs})

        for season, items in s2_index.items():
            logging.info(f"Season {season}: "  f"{len(items)} S2 composites")

        # PROCESS EACH S1 COMPOSITE
        for s1 in s1_files:
            logging.info("")
            logging.info(f"▶ Processing {s1.name}")

            out_mask = (MASK_DIR / f"{s1.stem}_habitat_mask.tif")

            try:
                with rasterio.open(s1) as src1:
                    geom1 = box(*src1.bounds)
                    season = s1.stem.split("_")[2]
                    logging.info( f"Season detected: {season}")

                    # CREATE MULTI-CLASS HABITAT MASK
                    if not out_mask.exists():
                        logging.info("▶ Creating habitat mask...")

                        # Select mangrove polygons intersecting S1
                        subset = mangrove_gdf[mangrove_gdf.intersects(geom1)]

                        # Mangrove binary mask
                        if len(subset) > 0:
                            mangrove_mask = (rasterio.features.rasterize([(geom, 1) for geom in subset.geometry], out_shape=(src1.height, src1.width),  transform=src1.transform, fill=0, dtype=np.uint8))

                        else:
                            mangrove_mask = np.zeros((src1.height, src1.width), dtype=np.uint8)

                        # Reproject habitat raster
                        # exactly onto S1 grid
                        habitat_reproj = np.zeros((src1.height,  src1.width),  dtype=np.uint8)
                        reproject(source=rasterio.band(habitat_src, 1), destination=habitat_reproj,
                            src_transform=(habitat_src.transform),
                            src_crs=habitat_src.crs,
                            dst_transform=src1.transform,
                            dst_crs=src1.crs,
                            resampling=(Resampling.nearest))

                        # Fusion:
                        # Keep habitat classes only inside
                        # reference mangrove polygons.
                        #
                        # Outside mangrove = 0
                        mask_full = np.where(mangrove_mask == 1, habitat_reproj, 0).astype(np.uint8)

                        # Save aligned habitat mask
                        profile = src1.profile.copy()
                        profile.update(count=1, dtype=rasterio.uint8, nodata=0, compress="LZW")

                        with rasterio.open(out_mask, "w", **profile) as dst:
                            dst.write(mask_full, 1)

                        logging.info(f"✔ Habitat mask saved: " f"{out_mask.name}")

                    else:
                        logging.info(f"✔ Existing habitat mask used: "  f"{out_mask.name}")


                    # FIND MATCHING S2 COMPOSITES
                    candidate_s2 = (s2_index.get(season, []))

                    if len(candidate_s2) == 0:
                        logging.warning(f"⚠ No S2 found for " f"season {season}")
                        continue


                    with rasterio.open(out_mask) as mask_src:
                        transform = src1.transform

                        # MATCH EACH INTERSECTING S2
                        for item in candidate_s2:
                            s2 = item["path"]
                            geom2 = item["geom"]

                            # No spatial overlap
                            if not geom1.intersects(geom2):
                                continue


                            inter = geom1.intersection(geom2)

                            # PATCH EXTRACTION
                            for y in range(0, src1.height - PATCH_SIZE + 1,  STRIDE):
                                for x in range(0,  src1.width  - PATCH_SIZE + 1,  STRIDE):
                                    w = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                                    bounds = (rasterio.windows.bounds(w, transform))

                                    # S1/S2 common extent
                                    if not box(*bounds).intersects(inter):
                                        continue

                                    # Read habitat mask
                                    mask_patch = (mask_src.read(1, window=w))

                                    # Pixels with habitat label : (< 5 & > 0) are considered valid
                                    # mangrove habitat pixels
                                    valid_habitat = ((mask_patch >= 1) & (mask_patch <= 4))
                                    n_habitat_pixels = int(valid_habitat.sum())

                                    
                                    # KEEP ONLY PATCHES CONTAINING
                                    # AT LEAST ONE LABELED MANGROVE
                                    # HABITAT PIXEL
                                    if n_habitat_pixels == 0:
                                        continue

                                    habitat_ratio = (n_habitat_pixels / mask_patch.size)

                                    # Classes actually present in this patch
                                    classes_present = np.unique(mask_patch[valid_habitat])

                                    # Patch geographic centre
                                    cx = (bounds[0] + bounds[2]) / 2
                                    cy = (bounds[1]  + bounds[3]) / 2


                                    rows.append({
                                            "tif": str(s1),
                                            "s2": str(s2),
                                            "mask": str(out_mask),
                                            "season": season,
                                            "x": int(x),
                                            "y": int(y),
                                            "xmin": bounds[0],
                                            "ymin": bounds[1],
                                            "xmax": bounds[2],
                                            "ymax": bounds[3],
                                            "cx": cx,
                                            "cy": cy,
                                            "n_habitat_pixels": n_habitat_pixels,
                                            "habitat_ratio": habitat_ratio,
                                            "n_classes": len(classes_present)})


            except Exception as e:
                logging.exception(f"⚠ Error for {s1.name}: {e}")


        # CREATE DATAFRAME
        df = pd.DataFrame(rows)

        if len(df) == 0:
            raise RuntimeError(
                "No habitat patches were created. "
                "Check S1/S2 season matching, spatial overlap "
                "and habitat labels.")

        # Reset clean index
        df = df.reset_index(drop=True)

        # BASIC SANITY CHECKS
        logging.info("")
        logging.info("===== BASIC INDEX CHECK =====")

        logging.info(f"Total retained patches: {len(df)}")
        logging.info(f"Unique S1 composites" f"{df['tif'].nunique()}")

        logging.info(f"Unique S2 composites: " f"{df['s2'].nunique()}")
        logging.info(f"Seasons: "  f"{sorted(df['season'].unique())}")

        logging.info(f"Mean habitat coverage: "  f"{100 * df['habitat_ratio'].mean():.2f}%")
        logging.info(f"Median habitat coverage: " f"{100 * df['habitat_ratio'].median():.2f}%")

        logging.info(f"Min habitat coverage: " f"{100 * df['habitat_ratio'].min():.4f}%")
        logging.info(f"Max habitat coverage: " f"{100 * df['habitat_ratio'].max():.2f}%")

        # SAVE TOTAL INDEX
        df.to_parquet(TOTAL_INDEX, index=False)

        logging.info(f"✔ Index created: {len(df)} samples")

    # Close habitat raster
    habitat_src.close()

# PATCH-LEVEL HABITAT AUDIT
logging.info("")
logging.info("▶ Computing habitat statistics for every patch...")

patch_stats = []

for idx, row in df.iterrows():
    with rasterio.open(row["mask"]) as src:
        mask = src.read(1, window=Window(int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE))

    # seules les valeurs 1, 2, 3, 4 sont des habitats valides
    # 255 = NoData
    # 0 = hors mangrove / fond
    valid = ((mask >= 1) & (mask <= 4))
    n_mangrove = int(valid.sum())
    n_total = int(mask.size)

    mangrove_ratio = n_mangrove / n_total

    # Classes habitat réellement présentes
    values, counts = np.unique(mask[valid], return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(values, counts)}
    classes_present = sorted(class_counts.keys())

    if len(class_counts) > 0:
        dominant_class = max(class_counts, key=class_counts.get)

    else:
        dominant_class = -1

    patch_stats.append({
        "n_mangrove_pixels": n_mangrove,
        "mangrove_ratio": mangrove_ratio,
        "classes_present": classes_present,
        "dominant_class": dominant_class,
        "class_counts": class_counts})


patch_stats_df = pd.DataFrame(patch_stats)
df = df.reset_index(drop=True)
patch_stats_df = patch_stats_df.reset_index(drop=True)

# ADD / REPLACE AUDIT INFORMATION
df["n_mangrove_pixels"] = patch_stats_df["n_mangrove_pixels"]
df["mangrove_ratio"] = patch_stats_df["mangrove_ratio"]

df["classes_present"] = patch_stats_df["classes_present"]
df["dominant_class"] = patch_stats_df["dominant_class"]
df["class_counts"] = patch_stats_df["class_counts"]

# Recalculate n_classes correctly:
# only classes 1, 2, 3, 4 are counted
df["n_classes"] = df["classes_present"].apply(len)
logging.info("✔ Patch habitat statistics computed")


logging.info("")
logging.info("===== PATCH CONTENT =====")

n_positive = (df["n_mangrove_pixels"] > 0).sum()
n_empty = (df["n_mangrove_pixels"] == 0).sum()

logging.info(f"Total patches              : {len(df)}")
logging.info(f"Patches containing habitat : {n_positive}")
logging.info(f"Empty patches              : {n_empty}")

logging.info(f"Mean mangrove coverage     : "  f"{100 * df['mangrove_ratio'].mean():.2f}%")
logging.info(f"Median mangrove coverage   : " f"{100 * df['mangrove_ratio'].median():.2f}%")
logging.info(f"Min mangrove coverage      : " f"{100 * df['mangrove_ratio'].min():.2f}%")
logging.info(f"Max mangrove coverage      : " f"{100 * df['mangrove_ratio'].max():.2f}%")


# GLOBAL CLASS DISTRIBUTION
global_counter = Counter()
for counts_dict in df["class_counts"]:
    for c, n in counts_dict.items():
        global_counter[c] += n

logging.info("")
logging.info("=" * 70)
logging.info("GLOBAL HABITAT DISTRIBUTION")
logging.info("=" * 70)

total_habitat_pixels = sum(global_counter.values())
habitat_stats = []
for c in sorted(global_counter):
    pixels = global_counter[c]
    patches_with_class = df[df["classes_present"].apply(lambda classes: c in classes)].shape[0]

    habitat_stats.append({
        "class": c,
        "pixels": pixels,
        "percent": (100 * pixels / total_habitat_pixels),
        "patches": patches_with_class,
        "patch_percent": (100 * patches_with_class / len(df))})


habitat_stats_df = pd.DataFrame(habitat_stats)
for _, r in habitat_stats_df.iterrows():
    logging.info(
        f"Class {int(r['class'])}: "
        f"{int(r['pixels']):,} pixels | "
        f"{r['percent']:.2f}% | "
        f"{int(r['patches'])} patches | "
        f"{r['patch_percent']:.2f}% of patches")

habitat_stats_df.to_csv(OUT_DIR / "habitat_global_distribution.csv", index=False)

logging.info("")
logging.info("===== NUMBER OF HABITATS PER PATCH =====")

class_number_distribution = (df["n_classes"] .value_counts() .sort_index())
for n_classes, n_patches in class_number_distribution.items():
    logging.info(f"{n_classes} habitat(s): " f"{n_patches} patches "  f"({100*n_patches/len(df):.2f}%)")

logging.info("")
logging.info("=" * 70)
logging.info("HABITAT DISTRIBUTION BY SEASON")
logging.info("=" * 70)

season_stats = []
for season in sorted(df["season"].unique()):
    subset = df[df["season"] == season]
    counter = Counter()

    for counts_dict in subset["class_counts"]:
        for c, n in counts_dict.items():
            counter[c] += n

    total = sum(counter.values())

    logging.info("")
    logging.info(f"===== {season} =====")
    logging.info(f"Patches: {len(subset)}")

    for c in sorted(global_counter):
        pixels = counter.get(c, 0)
        percent = (100 * pixels / total if total > 0  else 0)

        patches_with_class = subset[
            subset["classes_present"].apply(lambda classes: c in classes)].shape[0]

        logging.info(
            f"Class {c}: "
            f"{pixels:,} pixels | "
            f"{percent:.2f}% | "
            f"{patches_with_class} patches")

        season_stats.append({
            "season": season,
            "class": c,
            "pixels": pixels,
            "percent": percent,
            "patches": patches_with_class})


pd.DataFrame(season_stats).to_csv(OUT_DIR / "habitat_distribution_by_season.csv", index=False)

# SPATIAL LOCATION ID
SPATIAL_GROUP_SIZE = PATCH_SIZE * 10  # 1280 m
df["spatial_col"] = np.floor(df["cx"] / SPATIAL_GROUP_SIZE).astype(int)
df["spatial_row"] = np.floor(df["cy"] / SPATIAL_GROUP_SIZE).astype(int)
df["location_id"] = (df["spatial_col"].astype(str) + "_"  + df["spatial_row"].astype(str))

logging.info("")
logging.info(f"Distinct spatial patch groups: " f"{df['location_id'].nunique()}")

logging.info("")
logging.info("===== SPATIAL SUPPORT PER HABITAT =====")

for c in sorted(global_counter):
    subset = df[df["classes_present"].apply(lambda classes: c in classes)]
    n_locations = subset["location_id"].nunique()
    logging.info(f"Class {c}: " f"{n_locations} spatial patch groups")

AUDIT_INDEX = OUT_DIR / "habitat_audit_index.parquet"
audit_df = df.drop(columns=["class_counts", "classes_present"], errors="ignore").copy()
audit_df.to_parquet(AUDIT_INDEX, index=False)

logging.info("")
logging.info(f"✔ Audit index saved: {AUDIT_INDEX}")
logging.info("✔ Habitat dataset audit completed")

# CHECK EXACT SPATIAL PATCH LOCATIONS
df["patch_location_id"] = (
    df["xmin"].round(1).astype(str)
    + "_"
    + df["ymin"].round(1).astype(str)
    + "_"
    + df["xmax"].round(1).astype(str)
    + "_"
    + df["ymax"].round(1).astype(str))

logging.info("")
logging.info("===== SPATIAL GROUP CHECK =====")

logging.info(f"1280-m spatial groups       : " f"{df['location_id'].nunique()}")
logging.info(f"Exact patch locations       : " f"{df['patch_location_id'].nunique()}")


# Number of seasons available for each exact location
location_season_stats = (df.groupby("patch_location_id")["season"] .nunique() .value_counts() .sort_index())

logging.info("")
logging.info("===== SEASONS PER SPATIAL LOCATION =====")

for n_seasons, n_locations in location_season_stats.items():
    logging.info(f"{n_seasons} season(s): " f"{n_locations} spatial locations")


# GROUPED SPATIAL TRAIN / VALIDATION / TEST SPLIT
logging.info("")
logging.info("=" * 70)
logging.info("GROUPED SPATIAL TRAIN / VALIDATION / TEST SPLIT")
logging.info("=" * 70)

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

N_SPLIT_TRIALS = 5000
SPLIT_SEED = 42

CLASSES = [1, 2, 3, 4]

# BUILD ONE RECORD PER SPATIAL GROUP
logging.info("")
logging.info("▶ Building spatial-group statistics...")

group_records = []

for location_id, group in df.groupby("location_id"):
    class_pixels = {c: 0  for c in CLASSES}
    for counts_dict in group["class_counts"]:
        for c, n in counts_dict.items():
            c = int(c)
            if c in CLASSES:
                class_pixels[c] += int(n)

    group_records.append({
            "location_id": location_id,
            # Number of temporal observations / rows
            "n_samples": len(group),
            # Number of exact geographic patches
            "n_exact_locations": group["patch_location_id"].nunique(), **{f"class_{c}_pixels": class_pixels[c]  for c in CLASSES}})


groups_df = pd.DataFrame(group_records)
logging.info(f"Spatial groups available: {len(groups_df)}")

# GLOBAL TARGET DISTRIBUTION
global_class_pixels = np.array([groups_df[f"class_{c}_pixels"].sum() for c in CLASSES],  dtype=np.float64)
global_class_distribution = (global_class_pixels / global_class_pixels.sum())

logging.info("")
logging.info("Target habitat distribution:")

for c, p in zip(CLASSES, global_class_distribution):
    logging.info(f"Class {c}: {100 * p:.2f}%")


# FUNCTION TO EVALUATE A SPLIT
def evaluate_split(train_groups, val_groups, test_groups, groups_df):
    split_groups = {"train": train_groups,  "val": val_groups,  "test": test_groups}
    target_ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}

    score = 0.0
    total_samples = groups_df["n_samples"].sum()
    total_groups = len(groups_df)

    for split_name, ids in split_groups.items():
        subset = groups_df[groups_df["location_id"].isin(ids)]
        sample_ratio = (subset["n_samples"].sum() / total_samples)
        target_ratio = (target_ratios[split_name])
        score += (abs(sample_ratio - target_ratio) * 2.0)

        group_ratio = (len(subset)  / total_groups)
        score += (abs(group_ratio - target_ratio) * 1.0)

        class_pixels = np.array([subset[f"class_{c}_pixels"].sum() for c in CLASSES], dtype=np.float64)

        if class_pixels.sum() == 0:
            return np.inf

        distribution = (class_pixels / class_pixels.sum())
        score += (np.abs(distribution - global_class_distribution).mean() * 5.0)

        if np.any(class_pixels == 0):
            return np.inf

    return score

# RANDOM SEARCH FOR A GOOD GROUPED SPLIT
logging.info("")
logging.info(f"▶ Searching {N_SPLIT_TRIALS} candidate spatial splits...")

rng = np.random.RandomState(SPLIT_SEED)
group_ids = groups_df["location_id"].values.copy()

best_score = np.inf
best_split = None

for trial in range(N_SPLIT_TRIALS):
    shuffled = group_ids.copy()
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_train = int(round(TRAIN_RATIO * n_groups))
    n_val = int(round(VAL_RATIO * n_groups))

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train: n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    score = evaluate_split(train_ids, val_ids, test_ids, groups_df)

    if score < best_score:
        best_score = score
        best_split = {
            "train": set(train_ids),
            "val": set(val_ids),
            "test": set(test_ids)}

if best_split is None:
    raise RuntimeError("Unable to create a valid grouped split.")

logging.info(f"✔ Best split score: " f"{best_score:.6f}")


# ASSIGN SPLIT TO ORIGINAL DATAFRAME
def assign_split(location_id):
    if location_id in best_split["train"]:
        return "train"
    if location_id in best_split["val"]:
        return "val"
    if location_id in best_split["test"]:
        return "test"
    raise RuntimeError(f"Unknown location_id: {location_id}")


df["split"] = (df["location_id"] .apply(assign_split))
train_df = (df[df["split"] == "train"] .copy() .reset_index(drop=True))
val_df = (df[df["split"] == "val"] .copy() .reset_index(drop=True))
test_df = (df[df["split"] == "test"] .copy() .reset_index(drop=True))

# STRICT LEAKAGE CHECK
train_groups = set(train_df["location_id"])
val_groups = set(val_df["location_id"])
test_groups = set(test_df["location_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Also verify exact geographic patch locations
train_exact = set(train_df["patch_location_id"])
val_exact = set(val_df["patch_location_id"])
test_exact = set(test_df["patch_location_id"])

assert train_exact.isdisjoint(val_exact)
assert train_exact.isdisjoint(test_exact)
assert val_exact.isdisjoint(test_exact)

logging.info("✔ No spatial-group leakage detected")
logging.info("✔ No exact-location leakage detected")

# SPLIT AUDIT
logging.info("")
logging.info("=" * 70)
logging.info("FINAL SPLIT AUDIT")
logging.info("=" * 70)

for split_name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    logging.info("")
    logging.info(f"===== {split_name} =====")

    logging.info(f"Samples: " f"{len(split_df)} " f"({100 * len(split_df) / len(df):.2f}%)")
    logging.info(f"Spatial groups: " f"{split_df['location_id'].nunique()}")
    logging.info(f"Exact patch locations: " f"{split_df['patch_location_id'].nunique()}")

    # Seasons
    logging.info("Seasons: " + str(split_df["season"].value_counts().to_dict()))

    # Habitat distribution
    split_counter = Counter()

    for counts_dict in split_df["class_counts"]:
        for c, n in counts_dict.items():
            if int(c) in CLASSES:
                split_counter[int(c)] += int(n)

    total_pixels = sum(split_counter.values())
    logging.info("Habitat distribution:")

    for c in CLASSES:
        pixels = split_counter[c]
        percent = (100 * pixels / total_pixels  if total_pixels > 0  else 0)
        patches_with_class = (split_df["classes_present"].apply(lambda classes: c in classes).sum())

        locations_with_class = (split_df[split_df["classes_present"].apply(lambda classes: c in classes)]["location_id"].nunique())

        logging.info(
            f"  Class {c}: "
            f"{pixels:,} pixels | "
            f"{percent:.2f}% | "
            f"{patches_with_class} patches | "
            f"{locations_with_class} spatial groups")


# SAVE SPLITS
columns_to_drop = ["class_counts", "classes_present"]
train_save = train_df.drop(columns=columns_to_drop, errors="ignore")
val_save = val_df.drop(columns=columns_to_drop, errors="ignore")
test_save = test_df.drop(columns=columns_to_drop, errors="ignore")

TRAIN_INDEX = (OUT_DIR / "train_index.parquet")
VAL_INDEX = (OUT_DIR / "val_index.parquet")
TEST_INDEX = (OUT_DIR / "test_index.parquet")

train_save.to_parquet(TRAIN_INDEX, index=False)
val_save.to_parquet(VAL_INDEX, index=False)
test_save.to_parquet(TEST_INDEX, index=False)

logging.info("")
logging.info(f"✔ Train index saved: {TRAIN_INDEX}")
logging.info(f"✔ Validation index saved: {VAL_INDEX}")
logging.info(f"✔ Test index saved: {TEST_INDEX}")

logging.info("")
logging.info("✔ Grouped spatial split completed")

# DATASET
IGNORE_INDEX = 255
NUM_CLASSES = 4

class MangroveDataset(Dataset):
    def __init__(self, index_path):
        self.index = pd.read_parquet(index_path)
        logging.info(f"Dataset: {len(self.index)} samples | " f"{self.index['tif'].nunique()} S1")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def compute_indices(s2_patch):
        eps = 1e-6

        B03 = s2_patch[2]
        B04 = s2_patch[3]
        B05 = s2_patch[4]
        B06 = s2_patch[5]
        B08 = s2_patch[7]
        B8A = s2_patch[8]
        B11 = s2_patch[10]

        # NDVI
        ndvi = ((B08 - B04) / (B08 + B04 + eps))

        # NDMI
        ndmi = ((B08 - B11) / (B08 + B11 + eps))

        # Red-edge NDVI
        ndvi_re = ((B8A - B05) / (B8A + B05 + eps))

        # MERIS Terrestrial Chlorophyll Index : MTCI
        denom = B05 - B04
        mtci = np.zeros_like(denom, dtype=np.float32)
        valid_mtci = np.abs(denom) > 1.0
        mtci[valid_mtci] = ((B06[valid_mtci] - B05[valid_mtci]) / denom[valid_mtci])
        mtci = np.clip(mtci,-20.6206, 19.3479)

        indices = np.stack([mtci, ndmi, ndvi_re, ndvi], axis=0).astype(np.float32)

        # Protection against NaN / inf
        indices = np.nan_to_num(indices, nan=0.0, posinf=0.0, neginf=0.0)

        return indices

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        tif = row["tif"]
        s2 = row["s2"]
        mask_path = row["mask"]

        x = int(row["x"])
        y = int(row["y"])

        w = Window(x, y, PATCH_SIZE, PATCH_SIZE)

        try:
            # SENTINEL-1
            with rasterio.open(tif) as src1:
                s1_patch = src1.read(window=w).astype(np.float32)
                s1_transform = src1.transform
                s1_crs = src1.crs

                dst_transform = (rasterio.windows.transform(w, s1_transform))

            # SENTINEL-2
            with rasterio.open(s2) as src2:
                s2_resampled = np.zeros((src2.count, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                for i in range(src2.count):
                    reproject(source=rasterio.band(src2, i + 1),
                        destination=s2_resampled[i],
                        src_transform=src2.transform,
                        src_crs=src2.crs,
                        dst_transform=dst_transform,
                        dst_crs=s1_crs,
                        resampling=Resampling.bilinear)

                s2_patch = s2_resampled

            # HABITAT MASK
            with rasterio.open(mask_path) as m:
                mask_raw = m.read(1, window=w).astype(np.uint8)

            # REMAP LABELS
            #
            # raw:
            # 1,2,3,4 = habitats
            # 0       = outside mangrove
            # 255     = NoData
            #
            # PyTorch:
            # 0,1,2,3 = habitats
            # 255     = ignored

            mask = np.full(mask_raw.shape, IGNORE_INDEX, dtype=np.int64)
            valid = ((mask_raw >= 1) & (mask_raw <= 4))
            mask[valid] = (mask_raw[valid] - 1)

        except Exception as e:
            logging.warning(f"⚠ Patch error at idx={idx}: {e}")

            s1_patch = np.zeros((2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            s2_patch = np.zeros((12, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

            # IMPORTANT: on ignore entièrement un patch corrompu
            mask = np.full((PATCH_SIZE, PATCH_SIZE), IGNORE_INDEX, dtype=np.int64)

        # AUXILIARY INDICES
        # Sentinel-2 original resampled reflectances
        s2_raw = s2_patch.copy()

        # Compute indices from original reflectances
        indices = self.compute_indices(s2_raw)

        # NORMALISATION
        s1_patch = np.clip(s1_patch, -25, 5)
        s2_patch = (np.clip(s2_patch, 0, 10000) / 10000.0)

        # CONCATENATION
        patch = np.concatenate([s1_patch, s2_patch], axis=0)

        assert patch.shape[0] == 14
        assert indices.shape[0] == 4

        # TO TORCH
        patch = torch.tensor(patch, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        indices = torch.tensor(indices, dtype=torch.float32)

        return (patch, indices, mask, tif, x, y)

# DATALOADERS
train_dataset = MangroveDataset(TRAIN_INDEX)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

val_dataset = MangroveDataset(VAL_INDEX)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_dataset = MangroveDataset(TEST_INDEX)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# UPerNet Decoder for CROMA
class PPM(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, pool_scales=(1,2,3,6)):
        super().__init__()

        self.stages = nn.ModuleList()

        for s in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_channels, out_channels,1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales)*out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self,x):

        H,W = x.shape[-2:]
        pyramids = [x]

        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(H,W), mode="bilinear", align_corners=False)
            pyramids.append(y)

        x = torch.cat(pyramids,dim=1)
        return self.bottleneck(x)

class AuxiliaryIndexBranch(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.branch(x)

class UPerHead(nn.Module):

    def __init__(self, in_channels=256,  num_classes=NUM_CLASSES):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(self,x):
        x = self.conv(x)
        return self.cls(x)



class CROMA_UPerNet(nn.Module):

    def __init__(self, num_classes, aux_channels=64):
        super().__init__()

        # Encoder
        self.encoder = PretrainedCROMA(
            pretrained_path=str(BASE / "Mangrove_CROMA_Project/models/CROMA_base.pt"),
            size="base",
            modality="both",
            image_resolution=PATCH_SIZE)

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        # Fusion
        # SAR (768)
        # Optical (768)
        # Joint (768)

        self.proj = nn.Sequential(nn.Conv2d(768*3, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.ppm = PPM(in_channels=512, out_channels=256)

        self.aux_branch = AuxiliaryIndexBranch(in_channels=4, out_channels=aux_channels)
        self.fusion = nn.Sequential(nn.Conv2d(256 + aux_channels, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

        self.head = UPerHead(in_channels=256, num_classes=num_classes)

    def tokens_to_map(self,tokens):
        B,N,C = tokens.shape
        s = int(np.sqrt(N))
        return (tokens.reshape(B,s,s,C).permute(0,3,1,2))

    def forward(self, s1, s2, indices):

        out = self.encoder(SAR_images=s1, optical_images=s2)

        sar = self.tokens_to_map(out["SAR_encodings"])
        optical = self.tokens_to_map(out["optical_encodings"])
        joint = self.tokens_to_map(out["joint_encodings"])

        x = torch.cat([sar, optical, joint], dim=1)
        x = self.proj(x)
        x = self.ppm(x)
        aux = self.aux_branch(indices)
        aux = F.interpolate(aux, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, aux], dim=1)
        x = self.fusion(x)
        x = self.head(x)
        x = F.interpolate(x, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)

        return x


NUM_CLASSES = 4
model = CROMA_UPerNet(num_classes=NUM_CLASSES).to(DEVICE)

class OrdinalCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=255, ordinal_weight=0.2):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ordinal_weight = ordinal_weight

        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)


    def forward(self, logits, targets):

        # STANDARD CROSS ENTROPY
        ce_loss = self.ce(logits, targets)

        # VALID MANGROVE PIXELS ONLY
        valid = (targets != self.ignore_index)

        if not valid.any():
            return ce_loss

        # CLASS PROBABILITIES
        probs = torch.softmax(logits, dim=1)

        # [0, 1, 2, 3]
        class_values = torch.arange(self.num_classes, device=logits.device, dtype=probs.dtype).view(1, -1, 1, 1)

        # Expected ordinal stage
        expected_stage = (probs * class_values).sum(dim=1)

        # ORDINAL DISTANCE
        target_float = (targets.float())
        ordinal_error = torch.abs(expected_stage[valid] - target_float[valid])
        ordinal_loss = (ordinal_error.mean())

        # FINAL LOSS
        total_loss = (ce_loss + self.ordinal_weight  * ordinal_loss)
        return total_loss

criterion = OrdinalCrossEntropyLoss(num_classes=4, ignore_index=255, ordinal_weight=0.2)
optimizer = AdamW([{"params": model.proj.parameters(), "lr": 5e-4},
                   {"params": model.ppm.parameters(), "lr": 5e-4},
                   {"params": model.aux_branch.parameters(), "lr": 5e-4},
                   {"params": model.fusion.parameters(), "lr": 5e-4},
                   {"params": model.head.parameters(), "lr": 5e-4},], weight_decay=1e-4)


# PREFLIGHT CHECK
logging.info("▶ Running preflight check...")

torch.backends.cudnn.benchmark = True
use_amp = (DEVICE.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

patches, indices, masks, _, _, _ = next(iter(train_loader))

logging.info(f"patches shape : {patches.shape}")
logging.info(f"indices shape : {indices.shape}")
logging.info(f"masks shape   : {masks.shape}")

assert patches.ndim == 4
assert indices.ndim == 4
assert masks.ndim == 3

assert patches.shape[1] == 14, (f"Expected 14 input channels, " f"got {patches.shape[1]}")
assert indices.shape[1] == 4, (f"Expected 4 auxiliary indices, " f"got {indices.shape[1]}")
assert torch.isfinite(patches).all(), ("Non-finite values detected in patches")
assert torch.isfinite(indices).all(), ("Non-finite values detected in indices")

# DISPLAY INDEX STATISTICS
INDEX_NAMES = ["MTCI", "NDMI", "NDVI_RE", "NDVI"]

for j, name in enumerate(INDEX_NAMES):
    values = indices[:, j]
    logging.info(f"{name:8s} | "
        f"min={values.min().item():.4f} | "
        f"mean={values.mean().item():.4f} | "
        f"std={values.std().item():.4f} | "
        f"max={values.max().item():.4f}")

# FORWARD PASS
patches = patches.to(DEVICE, non_blocking=True)
indices = indices.to(DEVICE, non_blocking=True)
masks = masks.to(DEVICE, non_blocking=True)

s1 = patches[:, :2]
s2 = patches[:, 2:]

model.eval()
model.encoder.eval()

with torch.no_grad():
    logits = model(s1, s2, indices)

logging.info(f"S1 shape      : {s1.shape}")
logging.info(f"S2 shape      : {s2.shape}")
logging.info(f"indices shape : {indices.shape}")
logging.info(f"logits shape  : {logits.shape}")

assert logits.shape == (patches.shape[0], NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), (f"Unexpected logits shape: " f"{logits.shape}")
assert torch.isfinite(logits).all(), ("Non-finite values detected in logits")

# LOSS CHECK
loss_test = criterion(logits, masks)
assert torch.isfinite(loss_test), (f"Non-finite loss detected: {loss_test}")
logging.info(f"Preflight loss : {loss_test.item():.4f}")
logging.info("✔ Preflight check successful")

# FULL TRAIN INDEX STATISTICS
logging.info("▶ Computing auxiliary-index statistics on TRAIN...")

all_values = {"MTCI": [], "NDMI": [], "NDVI_RE": [], "NDVI": []}

with torch.no_grad():
    for _, batch_indices, _, _, _, _ in train_loader:
        for j, name in enumerate(INDEX_NAMES):
            vals = batch_indices[:, j].reshape(-1)

            # finite only
            vals = vals[torch.isfinite(vals)]
            all_values[name].append(vals.cpu())

for name in INDEX_NAMES:
    vals = torch.cat(all_values[name]).numpy()
    q001, q01, q50, q99, q999 = np.quantile(vals, [0.001, 0.01, 0.50, 0.99, 0.999])

    logging.info(
        f"{name:8s} | "
        f"min={vals.min():.4f} | "
        f"q0.1%={q001:.4f} | "
        f"q1%={q01:.4f} | "
        f"median={q50:.4f} | "
        f"q99%={q99:.4f} | "
        f"q99.9%={q999:.4f} | "
        f"max={vals.max():.4f} | "
        f"mean={vals.mean():.4f} | "
        f"std={vals.std():.4f}")

# TRAINING
CHECKPOINT = OUT_DIR / "checkpoint_croma_upernet_multiclass_aux_indices.pt"
BEST_MODEL = OUT_DIR / "best_croma_upernet_multiclass_aux_indices.pth"

start_epoch = 0
best_iou = -1.0

patience = 20
patience_counter = 0

train_losses = []
val_ious = []

if CHECKPOINT.exists():
    logging.info(f"▶ Loading checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = (checkpoint["epoch"] + 1)
    best_iou = checkpoint.get("best_iou", -1.0)
    patience_counter = checkpoint.get("patience_counter",0)
    train_losses = checkpoint.get("train_losses", [])
    val_ious = checkpoint.get("val_ious", [])

    logging.info(f"✔ Resume from epoch {start_epoch + 1}")
    logging.info(f"✔ Best validation mIoU so far: " f"{best_iou:.4f}")
    logging.info(f"✔ Patience counter: " f"{patience_counter}/{patience}")

for epoch in range(start_epoch,EPOCHS):
    logging.info(f"▶ Epoch {epoch + 1}/{EPOCHS}")

    model.train()
    model.encoder.eval()
    train_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # TRAIN
    for i, (patches, indices, masks, _, _, _) in enumerate(train_loader):
        if i % 50 == 0:
            logging.info(f"Batch {i}/{len(train_loader)}")

        patches = patches.to(DEVICE, non_blocking=True)
        indices = indices.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # CROMA inputs
        s1 = patches[:, :2]
        s2 = patches[:, 2:]

        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(s1, s2, indices)
            loss_full = criterion(logits, masks)
            loss = (loss_full / ACCUM_STEPS)

        scaler.scale(loss).backward()
        do_step = (((i + 1) % ACCUM_STEPS == 0) or ((i + 1) == len(train_loader)))


        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        train_loss += loss_full.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.no_grad():
        for patches, indices, masks, _, _, _ in val_loader:
            patches = patches.to(DEVICE, non_blocking=True)
            indices = indices.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            s1 = patches[:, :2]
            s2 = patches[:, 2:]


            with torch.autocast(device_type=DEVICE.type,  enabled=use_amp):
                logits = model(s1, s2, indices)

            preds = torch.argmax(logits, dim=1)
            valid = (masks != IGNORE_INDEX)

            for c in range(NUM_CLASSES):
                pred_c = ((preds == c) & valid)
                mask_c = ((masks == c) & valid)

                intersections[c] += (pred_c & mask_c).sum().item()
                unions[c] += (pred_c | mask_c).sum().item()

    class_ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=(unions > 0))
    valid_classes = (unions > 0)
    val_iou = (class_ious[valid_classes].mean()  if valid_classes.any()  else 0.0)

    train_losses.append(train_loss)
    val_ious.append(val_iou)

    logging.info(f"Train Loss : {train_loss:.4f}")

    for c, iou in enumerate(class_ious):
        logging.info(f"Val IoU class {c}: " f"{iou:.4f}")

    logging.info(f"Val mIoU   : {val_iou:.4f}")

    # EARLY STOPPING
    if val_iou > best_iou + 1e-4:

        best_iou = val_iou
        patience_counter = 0

        torch.save(model.state_dict(), BEST_MODEL)
        logging.info("✔ Best model saved")

    else:

        patience_counter += 1
        logging.info(f"No improvement |" f"patience "  f"{patience_counter}/{patience}")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou": best_iou,
        "patience_counter": patience_counter,
        "train_losses": train_losses,
        "val_ious": val_ious,
    }, CHECKPOINT)

    logging.info(f"✔ Checkpoint saved | " f"epoch {epoch + 1}")


    if patience_counter >= patience:
        logging.info(f"Early stopping at epoch " f"{epoch + 1}")
        break

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# TEST EVALUATION
logging.info("▶ TESTING CROMA + UPerNet")

model = CROMA_UPerNet(num_classes=NUM_CLASSES).to(DEVICE)

if not BEST_MODEL.exists():
    raise FileNotFoundError(f"Best model not found: {BEST_MODEL}")

logging.info(f"▶ Loading best model: {BEST_MODEL}")

state_dict = torch.load(BEST_MODEL, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

logging.info(f"✔ Best model loaded | "  f"Best validation mIoU = {best_iou:.4f}")
logging.info("▶ Computing test metrics...")

tp = np.zeros(NUM_CLASSES, dtype=np.float64)
fp = np.zeros(NUM_CLASSES, dtype=np.float64)
fn = np.zeros(NUM_CLASSES, dtype=np.float64)

conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
distance_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

ordinal_abs_error = 0.0
n_valid_pixels = 0


with torch.no_grad():
    for patches, indices, masks, _, _, _ in test_loader:
        patches = patches.to(DEVICE)
        indices = indices.to(DEVICE)
        masks = masks.to(DEVICE)
        s1 = patches[:, :2]
        s2 = patches[:, 2:]

        logits = model(s1, s2, indices)
        preds = torch.argmax(logits, dim=1)
        valid = (masks != IGNORE_INDEX)

        true_valid = masks[valid]
        pred_valid = preds[valid]

        cm_indices = (true_valid * NUM_CLASSES + pred_valid)
        cm = torch.bincount(cm_indices, minlength=NUM_CLASSES * NUM_CLASSES)
        cm = (cm.reshape(NUM_CLASSES, NUM_CLASSES) .cpu() .numpy())
        conf_matrix += cm

        distances = torch.abs(pred_valid - true_valid)
        for d in range(NUM_CLASSES):
            distance_counts[d] += (distances == d).sum().item()

        # Ordinal MAE
        ordinal_abs_error += (torch.abs(preds[valid].float() - masks[valid].float()) .sum() .item())
        n_valid_pixels += (valid.sum().item())

        # Per-class statistics
        for c in range(NUM_CLASSES):
            pred_c = ((preds == c)  & valid)
            true_c = ((masks == c) & valid)

            tp[c] += (pred_c & true_c).sum().item()
            fp[c] += (pred_c & ~true_c & valid).sum().item()
            fn[c] += (~pred_c & true_c & valid).sum().item()

# FINAL METRICS
precision = (tp / np.maximum(tp + fp, 1))
recall = (tp / np.maximum(tp + fn, 1))
f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1))
iou = (tp / np.maximum(tp + fp + fn, 1))

mean_iou = iou.mean()
macro_f1 = f1.mean()

ordinal_mae = (ordinal_abs_error / max(n_valid_pixels, 1))
CLASS_NAMES = ["Young", "Adult", "Mature",  "Senescent"]

logging.info("")
logging.info("===== TEST RESULTS =====")

for c, name in enumerate(CLASS_NAMES):
    logging.info(
        f"{name:10s} | "
        f"Precision={precision[c]:.4f} | "
        f"Recall={recall[c]:.4f} | "
        f"F1={f1[c]:.4f} | "
        f"IoU={iou[c]:.4f}")

logging.info(f"Macro F1    : {macro_f1:.4f}")
logging.info(f"Mean IoU    : {mean_iou:.4f}")
logging.info(f"Ordinal MAE : {ordinal_mae:.4f}")

logging.info("")
logging.info("===== CONFUSION MATRIX =====")

logging.info("Rows = Ground Truth | " "Columns = Prediction")
logging.info(f"Classes = {CLASS_NAMES}")

for i, name in enumerate(CLASS_NAMES):
    logging.info(f"{name:10s}: "  f"{conf_matrix[i].tolist()}")

# Row-normalized confusion matrix
row_sum = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_norm = np.divide(conf_matrix, row_sum, out=np.zeros_like(conf_matrix, dtype=np.float64), where=row_sum != 0)

logging.info("")
logging.info("===== NORMALIZED CONFUSION MATRIX =====")

for i, name in enumerate(CLASS_NAMES):
    logging.info(f"{name:10s}: "  + " | ".join(f"{v:.2%}" for v in conf_matrix_norm[i]))

logging.info("")
logging.info("===== ORDINAL ERROR DISTRIBUTION =====")

total = distance_counts.sum()

for d, count in enumerate(distance_counts):
    logging.info(
        f"Distance {d}: "
        f"{count:,} pixels "
        f"({100 * count / max(total, 1):.2f}%)")
