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
import torch.nn.functional as F
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

os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"

BASE = Path(os.environ["WORK"])
BASE2 = Path(os.environ["SCRATCH"])

# GUYANE
S1_GUYANE = (BASE / "Mangrove_CROMA_Project/data/processed/S1_zone_composites_seasonal")
S2_GUYANE = (BASE / "Mangrove_CROMA_Project/data/processed/S2_seasonal_composite")

# SURINAME
S1_SURINAME = (BASE2 / "Mangrove_CROMA_Project/data/processed/S1_suriname_composites")
S2_SURINAME = (BASE2 / "Mangrove_CROMA_Project/data/processed/S2_suriname_seasonal_composites")

# AMAPA
S1_AMAPA = (BASE2 / "Mangrove_CROMA_Project/data/processed/S1_amapa_composites")
S2_AMAPA = (BASE2 / "Mangrove_CROMA_Project/data/processed/S2_amapa_seasonal_composites")

# MANGROVE VECTOR
MANGROVE_VECTOR = BASE /  "Mangrove_CROMA_Project/data/ancillary/vect_mang_Amapa_Guyana_GMW_2017_2020_2023/vect_mang_Amapa_Guyana_GMW_2023.shp"

OUT_DIR = BASE2 / "Mangrove_CROMA_Project/data/segmentation_binaire_croma_upernet_guyane_suriname_amapa"
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
        logging.FileHandler(LOG_DIR / "segmentation_fusion_s1-s2.log"),
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
LR = 1e-4
NUM_WORKERS = 2
PIN_MEMORY = True
POS_THRESHOLD = 0.01
NEG_KEEP_RATIO = 0.1

if IS_MAIN_PROCESS:
    # LOAD MANGROVES
    logging.info("▶ Loading mangrove polygons...")
    mangrove_gdf = gpd.read_file(MANGROVE_VECTOR)
    mangrove_gdf = mangrove_gdf.to_crs("EPSG:32622")
    mangrove_gdf = mangrove_gdf[mangrove_gdf.geometry.notnull()]
    mangrove_gdf = mangrove_gdf[mangrove_gdf.geometry.is_valid]

    index_path = TOTAL_INDEX

    if index_path.exists():
        try:
            df = pd.read_parquet(index_path)
            logging.info(f" Index loaded: {len(df)} samples")
        except:
            logging.warning("Corrupted index → rebuilding")
            index_path.unlink()
    else:
        logging.info("▶ Creating masks + fusion S1/S2 index...")

        rows = []
        rng = np.random.RandomState(SEED)

        s1_files = []
        for f in sorted(S1_GUYANE.glob("*.tif")):
            s1_files.append({"path": f, "region": "guyane"})

        for f in sorted(S1_SURINAME.glob("*.tif")):
            s1_files.append({"path": f, "region": "suriname"})

        for f in sorted(S1_AMAPA.glob("*.tif")):
            s1_files.append({"path": f, "region": "amapa"})

        s2_files = []
        s2_files.extend(sorted(S2_GUYANE.glob("*.tif")))
        s2_files.extend(sorted(S2_SURINAME.glob("*.tif")))
        s2_files.extend(sorted(S2_AMAPA.glob("*.tif")))

        # INDEX DES COMPOSITES S2 PAR SAISON
        s2_index = {}

        for s2 in s2_files:
            season = s2.stem.split("_")[-2]

            with rasterio.open(s2) as src:
                geom = box(*src.bounds)

            s2_index.setdefault(season, []).append({"path": s2, "geom": geom})

        logging.info(f"S1 files: {len(s1_files)}")
        logging.info(f"S2 files: {len(s2_files)}")

        # LOOP S1 : MASK + INTERSECTION
        for item_s1 in s1_files:
            s1 = item_s1["path"]
            region = item_s1["region"]
            out_mask = MASK_DIR / f"{s1.stem}_mask.tif"

            try:
                with rasterio.open(s1) as src1:

                    geom1 = box(*src1.bounds)

                    # CREATE MASK 
                    if not out_mask.exists():
                        logging.info(f"Creating mask: {out_mask.name}")

                        subset = mangrove_gdf[mangrove_gdf.intersects(geom1)]

                        if len(subset) > 0:
                            mask_full = rasterio.features.rasterize(
                                [(geom, 1) for geom in subset.geometry],
                                out_shape=(src1.height, src1.width),
                                transform=src1.transform,
                                fill=0,
                                dtype=np.uint8)
                            
                        else:
                            mask_full = np.zeros((src1.height, src1.width), dtype=np.uint8)

                        profile = src1.profile.copy()
                        profile.pop("nodata", None)
                        profile.update(count=1, dtype=rasterio.uint8, nodata=0)
                        print(s1.name, "nodata =", src1.nodata)

                        with rasterio.open(out_mask, "w", **profile) as dst:
                            dst.write(mask_full, 1)

                    # FIND MATCHING S2
                    candidate_s2 = s2_index.get(season, [])

                    for item in candidate_s2:
                        s2 = item["path"]
                        geom2 = item["geom"]

                        if not geom1.intersects(geom2):
                            continue

                        inter = geom1.intersection(geom2)
                        transform = src1.transform

                        for y in range(0, src1.height - PATCH_SIZE + 1, STRIDE):
                            for x in range(0, src1.width - PATCH_SIZE + 1, STRIDE):
                                w = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                                bounds = rasterio.windows.bounds(w, transform)

                                if not box(*bounds).intersects(inter):
                                    continue

                                with rasterio.open(out_mask) as m:
                                    mask_patch = m.read(1, window=w)

                                r = np.mean(mask_patch > 0)
                                if region == "guyane":
                                    keep = (r > 0 or rng.rand() < NEG_KEEP_RATIO)

                                else:
                                    keep = (r > 0)

                                if keep:
                                    cx = (bounds[0] + bounds[2]) / 2
                                    cy = (bounds[1] + bounds[3]) / 2
                                    rows.append({
                                        "region": region,
                                        "tif": str(s1),
                                        "s2": str(s2),
                                        "mask": str(out_mask),
                                        "season": season,
                                        "x": x,
                                        "y": y,
                                        "xmin": bounds[0],
                                        "ymin": bounds[1],
                                        "xmax": bounds[2],
                                        "ymax": bounds[3],
                                        "cx": cx,
                                        "cy": cy
                                    })

            except Exception as e:
                logging.warning(f"⚠ Error for {s1.name}: {e}")

        df = pd.DataFrame(rows)
        logging.info(f" Total patches: {len(df)}")

        df.to_parquet(index_path, index=False)
        logging.info(" Index saved")



logging.info("CHECKING S1/S2 PAIRS")

assoc = (df.groupby("tif")["s2"].nunique().sort_values(ascending=False))
logging.info(f"Min S2 per S1 : {assoc.min()}")
logging.info(f"Mean S2 per S1 : {assoc.mean():.2f}")
logging.info(f"Max S2 per S1 : {assoc.max()}")

for tif, n in assoc.items():
    logging.info(f"{Path(tif).stem}: {n}")

with rasterio.open(df.iloc[0]["tif"]) as s1:
    logging.info(f"S1 CRS={s1.crs} "  f"RES={s1.res}")

with rasterio.open(df.iloc[0]["s2"]) as s2:
    logging.info(f"S2 CRS={s2.crs} " f"RES={s2.res}")

for tif in df["tif"].unique():
    sub = df[df["tif"] == tif]

    logging.info("")
    logging.info(Path(tif).stem)
    logging.info(sub["s2"].value_counts().to_string())

# GRID SPLIT
logging.info("▶ Creating spatial grid...")

# SPLIT UNIQUEMENT SUR LA GUYANE
guyane_df = df[df["region"] == "guyane"].copy()
suriname_df = df[df["region"] == "suriname"].copy()
amapa_df = df[df["region"] == "amapa"].copy()

logging.info(f"Guyane patches   : {len(guyane_df)}")
logging.info(f"Suriname patches : {len(suriname_df)}")
logging.info(f"Amapa patches    : {len(amapa_df)}")

# GRILLE UNIQUEMENT SUR LA GUYANE
xmin = guyane_df["xmin"].min()
xmax = guyane_df["xmax"].max()

ymin = guyane_df["ymin"].min()
ymax = guyane_df["ymax"].max()

N_COLS = 7
N_ROWS = 4

dx = (xmax - xmin) / N_COLS
dy = (ymax - ymin) / N_ROWS

cells = []
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
for r in range(N_ROWS):
    for c in range(N_COLS):
        x0 = xmin + c * dx
        x1 = x0 + dx
        y0 = ymin + r * dy
        y1 = y0 + dy
        cells.append({
                "cell": f"{letters[r]}{c+1}",
                "geometry": box(x0, y0, x1, y1)})

with rasterio.open(
    guyane_df.iloc[0]["tif"]) as src:
    raster_crs = src.crs

grid = gpd.GeoDataFrame(cells, crs=raster_crs)

logging.info("✅ Grid created")

def assign_cell(row):
    pt = Point(row.cx, row.cy)
    m = grid[grid.contains(pt)]
    return (m.iloc[0]["cell"] if len(m) > 0  else "UNK")

guyane_df["cell"] = (guyane_df.apply(assign_cell, axis=1))

TRAIN = ["D1","C1","C2", "C3"]
VAL = ["C4", "B4"]
TEST = ["B5"]

def split(cell):

    if cell in TRAIN:
        return "train"

    elif cell in VAL:
        return "val"

    elif cell in TEST:
        return "test"

    else:
        return "exclude"

guyane_df["split"] = (guyane_df["cell"].apply(split))
guyane_df = guyane_df[guyane_df["split"].isin(["train", "val", "test"])].copy()

# SURINAME + AMAPA TOUJOURS DANS LE TRAIN
suriname_df["split"] = "train"
suriname_df["cell"] = "SURINAME"

amapa_df["split"] = "train"
amapa_df["cell"] = "AMAPA"

# CONCAT FINAL
df = pd.concat([guyane_df, suriname_df, amapa_df],ignore_index=True)

logging.info(" Final split created")
logging.info("\n" + str(df.groupby(["region", "split"]).size()))

df[df["split"]=="train"][["tif","s2","mask","x","y"]].to_parquet(TRAIN_INDEX, index=False)
df[df["split"]=="val"][["tif","s2","mask","x","y"]].to_parquet(VAL_INDEX, index=False)
df[df["split"]=="test"][["tif","s2","mask","x","y"]].to_parquet(TEST_INDEX, index=False)

logging.info(" Saved parquet files")

logging.info("\n===== REGION / SPLIT =====")
for (region, split_name), n in (df.groupby(["region", "split"]).size().items()):
    logging.info(f"{region:10s} | "  f"{split_name:5s} | "  f"{n}")

logging.info("CHECKING S1/S2 PAIRS")
assoc = (df.groupby("tif")["s2"].nunique().sort_values(ascending=False))
logging.info(f"Min S2 per S1 : {assoc.min()}")
logging.info(f"Mean S2 per S1 : {assoc.mean():.2f}")
logging.info(f"Max S2 per S1 : {assoc.max()}")

for tif, n in assoc.items():
    logging.info(f"{Path(tif).stem}: {n}")

with rasterio.open(df.iloc[0]["tif"]) as s1:
    logging.info(
        f"S1 CRS={s1.crs} "
        f"RES={s1.res}")

with rasterio.open(df.iloc[0]["s2"]) as s2:
    logging.info(
        f"S2 CRS={s2.crs} "
        f"RES={s2.res}")

# SPLIT DIAGNOSTICS
logging.info("▶ Computing split statistics...")
for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name]

    logging.info("")
    logging.info(f"===== {split_name.upper()} =====")
    logging.info(f"Patches : {len(subset)}")
    logging.info(f"S1 files: {subset['tif'].nunique()}")
    logging.info(f"S2 files: {subset['s2'].nunique()}")

    total_pixels = 0
    mangrove_pixels = 0

    positive_patches = 0
    negative_patches = 0

    for _, row in subset.iterrows():
        with rasterio.open(row["mask"]) as src:
            w = Window(int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE)
            mask = src.read(1, window=w)

        ratio = mask.mean()
        if ratio > 0:
            positive_patches += 1
        else:
            negative_patches += 1

        mangrove_pixels += mask.sum()
        total_pixels += mask.size

    non_mangrove_pixels = total_pixels - mangrove_pixels

    logging.info(f"Positive patches : {positive_patches}")
    logging.info(f"Negative patches : {negative_patches}")

    logging.info(f"Positive patches (%) : "  f"{100*positive_patches/len(subset):.2f}")
    logging.info(f"Negative patches (%) : "  f"{100*negative_patches/len(subset):.2f}")
    logging.info(f"Mangrove pixels (%) : "  f"{100*mangrove_pixels/total_pixels:.4f}")
    logging.info(f"Non-mangrove pixels (%) : " f"{100*non_mangrove_pixels/total_pixels:.4f}")

logging.info(" Split diagnostics done")

# CELL DISTRIBUTION
logging.info("")
logging.info("===== CELL DISTRIBUTION =====")

for cell in sorted(df["cell"].unique()):
    sub = df[df["cell"] == cell]
    logging.info(
        f"{cell:>3} | "
        f"{len(sub):5d} patches | "
        f"split={sub['split'].iloc[0]}")

for cell in sorted(df["cell"].unique()):

    subset = df[df["cell"] == cell]
    pos = 0

    for _, row in subset.iterrows():
        with rasterio.open(row["mask"]) as src:
            w = Window(int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE)
            mask = src.read(1, window=w)

        if mask.sum() > 0:
            pos += 1

    logging.info(f"{cell}: {pos}/{len(subset)} positive patches")


# Pourcentage de patches positifs par cellule
for cell in sorted(df["cell"].unique()):
    subset = df[df["cell"] == cell]
    pos = 0

    for _, row in subset.iterrows():
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1, window=Window( int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE))

        pos += mask.sum()

    ratio = pos / (len(subset) * PATCH_SIZE * PATCH_SIZE)
    logging.info(f"{cell}: {100*ratio:.2f}%")


# POSITIVE ONLY DATASETS
logging.info("")
logging.info("▶ Creating positive-only parquet files...")

positive_train = None
positive_val = None
positive_test = None

for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name]
    rows_keep = []

    for idx, row in subset.iterrows():
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1, window=Window( int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE))

        if mask.sum() > 0:
            rows_keep.append(idx)

    subset_pos = subset.loc[rows_keep].copy()
    logging.info(f"{split_name}: " f"{len(subset_pos)}/{len(subset)} positive patches kept")

    if split_name == "train":
        positive_train = subset_pos

    elif split_name == "val":
        positive_val = subset_pos

    elif split_name == "test":
        positive_test = subset_pos

TRAIN_PPOS = TRAIN_INDEX.parent / "train_ppos.parquet"
VAL_PPOS = VAL_INDEX.parent / "val_ppos.parquet"
TEST_PPOS = TEST_INDEX.parent / "test_ppos.parquet"

positive_train.to_parquet(TRAIN_PPOS, index=False)
positive_val.to_parquet(VAL_PPOS, index=False)
positive_test.to_parquet(TEST_PPOS, index=False)

logging.info(" Positive-only parquet files saved")
logging.info(f"TRAIN_PPOS : {TRAIN_PPOS}")
logging.info(f"VAL_PPOS   : {VAL_PPOS}")
logging.info(f"TEST_PPOS  : {TEST_PPOS}")


# POSITIVE-ONLY DIAGNOSTICS
for name, subset in [("TRAIN_PPOS", positive_train), ("VAL_PPOS", positive_val), ("TEST_PPOS", positive_test)]:
    logging.info("")
    logging.info(f"===== {name} =====")

    logging.info(f"Patches : {len(subset)}")
    logging.info(f"S1 files : {subset['tif'].nunique()}")
    logging.info(f"S2 files : {subset['s2'].nunique()}")

    total_pixels = 0
    mangrove_pixels = 0

    for _, row in subset.iterrows():
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1, window=Window(int(row["x"]), int(row["y"]), PATCH_SIZE, PATCH_SIZE))

        mangrove_pixels += mask.sum()
        total_pixels += mask.size

    ratio = (100 * mangrove_pixels / total_pixels if total_pixels > 0 else 0)
    logging.info(f"Mangrove pixels (%) : {ratio:.4f}")


# DATASET
class MangroveDataset(Dataset):

    def __init__(self, index_path, augment=False):
        self.index = pd.read_parquet(index_path)
        self.augment = augment

        logging.info(f"Dataset: {len(self.index)} samples | " f"{self.index['tif'].nunique()} S1")

    def __len__(self):
        return len(self.index)

    def apply_augmentation(self, patch, mask):

        # Flip horizontal
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Flip vertical
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Rotation 0°, 90°, 180°, 270°
        k = np.random.randint(4)
        if k > 0:
            patch = np.rot90(patch, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()

        return patch, mask

    def __getitem__(self, idx):

        row = self.index.iloc[idx]
        tif = row["tif"]
        s2  = row["s2"]
        mask_path = row["mask"]

        x = int(row["x"])
        y = int(row["y"])

        w = Window(x, y, PATCH_SIZE, PATCH_SIZE)

        try:
            # S1 
            with rasterio.open(tif) as src1:
                s1_patch = src1.read(window=w).astype(np.float32)

            # S2 ALIGNÉ
            with rasterio.open(s2) as src2:

                # créer patch S2 aligné
                s2_resampled = np.zeros((src2.count, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

                # transform du patch S1
                dst_transform = rasterio.windows.transform(w, src1.transform)

                for i in range(src2.count):
                    reproject(
                        source=rasterio.band(src2, i + 1),
                        destination=s2_resampled[i],
                        src_transform=src2.transform,
                        src_crs=src2.crs,
                        dst_transform=dst_transform,
                        dst_crs=src1.crs,
                        resampling=Resampling.bilinear)

                s2_patch = s2_resampled

            # MASK 
            with rasterio.open(mask_path) as m:
                mask = m.read(1, window=w).astype(np.float32)

        except Exception as e:
            logging.warning(f"⚠ Patch error: {e}")

            s1_patch = np.zeros((2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            s2_patch = np.zeros((12, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        # NORMALISATION
        s1_patch = np.clip(s1_patch, -25, 5)
        s2_patch = np.clip(s2_patch, 0, 10000) / 10000.0

        # CONCAT 
        s1_patch = np.nan_to_num(s1_patch, nan=-25.0)
        s2_patch = np.nan_to_num(s2_patch, nan=0.0)
        mask = np.nan_to_num(mask, nan=0.0)
        patch = np.concatenate([s1_patch, s2_patch], axis=0)

        if np.isnan(s1_patch).any():
            print("NaN S1")

        if np.isnan(s2_patch).any():
            print("NaN S2")

        if np.isnan(mask).any():
            print("NaN MASK")


        # DATA AUGMENTATION
        if self.augment:
            patch, mask = self.apply_augmentation(patch, mask)

        patch = torch.tensor(patch, dtype=torch.float32)
        mask  = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if np.isnan(patch).any():
            logging.warning("NaN detected in patch")

        if np.isnan(mask).any():
            logging.warning("NaN detected in mask")

        return patch, mask, tif, x, y

# DATALOADERS
train_dataset = MangroveDataset(TRAIN_PPOS, augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

val_dataset = MangroveDataset(VAL_PPOS,  augment=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_dataset = MangroveDataset(TEST_PPOS,  augment=False)
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


class UPerHead(nn.Module):

    def __init__(self,in_channels=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(256,1,1)

    def forward(self,x):
        x = self.conv(x)
        return self.cls(x)


class CROMA_UPerNet(nn.Module):

    def __init__(self):
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
        self.head = UPerHead(in_channels=256)

    def tokens_to_map(self,tokens):
        B,N,C = tokens.shape
        s = int(np.sqrt(N))
        return (tokens.reshape(B,s,s,C).permute(0,3,1,2))

    def forward(self,x):
        s1 = x[:,:2]
        s2 = x[:,2:]
        out = self.encoder(SAR_images=s1, optical_images=s2)

        # récupérer les trois représentations
        sar = self.tokens_to_map(out["SAR_encodings"])
        optical = self.tokens_to_map(out["optical_encodings"])
        joint = self.tokens_to_map(out["joint_encodings"])

        # fusion
        x = torch.cat([sar,optical,joint], dim=1)
        x = self.proj(x)
        x = self.ppm(x)
        x = self.head(x)
        x = F.interpolate(x, size=(PATCH_SIZE,PATCH_SIZE), mode="bilinear", align_corners=False)

        return x


model = CROMA_UPerNet().to(DEVICE)

# LOSS
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=8.0, smooth=1e-6):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=DEVICE))

    def forward(self, logits, targets):

        # FOCAL BCE
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_bce = (focal_weight * bce).mean()

        # DICE
        B = probs.shape[0]
        probs = probs.reshape(B, -1)
        targets = targets.reshape(B, -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return focal_bce + dice_loss


criterion = FocalDiceLoss(alpha=0.5, gamma=2, pos_weight=8)
optimizer = AdamW([
        {"params": model.proj.parameters(), "lr": 5e-4},
        {"params": model.ppm.parameters(), "lr": 5e-4},
        {"params": model.head.parameters(), "lr": 5e-4},
    ],
    weight_decay=1e-4)


# TRAINING
CHECKPOINT = OUT_DIR / "checkpoint.pt"

start_epoch = 0
best_iou = -1

patience = 20
patience_counter = 0

torch.backends.cudnn.benchmark = True

use_amp = (DEVICE.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if CHECKPOINT.exists():
    logging.info(f"Loading checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT,map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_iou = checkpoint["best_iou"]
    patience_counter = checkpoint.get("patience_counter", 0)

    logging.info(f"Resume from epoch {start_epoch}")
    logging.info(f"Best IoU so far: {best_iou:.4f}")


train_losses = []
val_ious = []

# TRAIN LOOP
for epoch in range(start_epoch, EPOCHS):
    logging.info(f"▶ Epoch {epoch+1}/{EPOCHS}")

    # TRAIN 
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for i, (patches, masks, _, _, _) in enumerate(train_loader):
        if i % 50 == 0:
            logging.info(f"Batch {i}/{len(train_loader)}")

        patches = patches.to(DEVICE, non_blocking=True)
        masks   = masks.to(DEVICE,  non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            preds = model(patches)
            loss  = criterion(preds, masks)

        loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()

        # CLEAN
        del patches, masks, preds, loss

    train_loss *= ACCUM_STEPS
    train_loss /= len(train_loader)

    # VALIDATION 
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for patches, masks, _, _, _ in val_loader:

            patches = patches.to(DEVICE)
            masks   = masks.to(DEVICE)

            preds = model(patches)
            preds = (torch.sigmoid(preds) > 0.5).int()

            tp += ((preds == 1) & (masks == 1)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()

            del patches, masks, preds

    val_iou = tp / (tp + fp + fn + 1e-6)

    # LOG 
    train_losses.append(train_loss)
    val_ious.append(val_iou)

    logging.info(f"Train Loss : {train_loss:.4f}")
    logging.info(f"Val IoU    : {val_iou:.4f}")

    # EARLY STOPPING
    if val_iou > best_iou + 1e-4:
        best_iou = val_iou
        patience_counter = 0

        torch.save(model.state_dict(), OUT_DIR / "best_model_croma_upernet.pth")
        torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou": best_iou,
        "patience_counter": patience_counter,},OUT_DIR / "checkpoint.pt")
        logging.info("✔ Best model saved")

    else:
        patience_counter += 1

    if patience_counter >= patience:
        logging.info(f" Early stopping at epoch {epoch+1}")
        break

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# PLOTS
thresholds = [0.5, 0.55, 0.6, 0.65]

BASE_OUT = BASE2 / OUT_DIR / "outputs_s1_s2"
BASE_OUT.mkdir(parents=True, exist_ok=True)

experiment_name = "UPerNet_CROMA_S1S2"

try:
    epochs_range = range(1, len(train_losses) + 1)

    curve_dir = BASE_OUT / f"{experiment_name}_curves"
    curve_dir.mkdir(exist_ok=True)

    # LOSS
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, train_losses, marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(curve_dir / "train_loss.png", dpi=300)
    plt.close()

    # IOU
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, val_ious, marker="o")
    plt.title("Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.savefig(curve_dir / "val_iou.png", dpi=300)
    plt.close()

    logging.info(" Plots saved")

except Exception as e:
    logging.warning(f"Plotting failed: {e}")


# TEST EVALUATION
logging.info("▶ TESTING S1 + S2")

# LOAD MODEL
model = CROMA_UPerNet().to(DEVICE)
state_dict = torch.load(OUT_DIR / "best_model_croma_upernet.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

thresholds = [0.5, 0.55, 0.6, 0.65]

BASE_OUT = BASE2 / "outputs_s1_s2"
BASE_OUT.mkdir(parents=True, exist_ok=True)

experiment_name = "UPerNet_CROMA_S1S2"
test_df = pd.read_parquet(TEST_PPOS)

# TEST LOOP
for threshold in thresholds:
    logging.info(f"\n--- Threshold {threshold} ---")

    exp_dir = BASE_OUT / f"{experiment_name}_th{threshold}"
    exp_dir.mkdir(exist_ok=True)

    VIS_DIR = exp_dir / "visualizations"
    VIS_DIR.mkdir(exist_ok=True)

    RASTER_DIR = exp_dir / "predicted_rasters"
    RASTER_DIR.mkdir(exist_ok=True)

    tp = fp = fn = tn = 0

    # METRICS
    with torch.no_grad():
        first_batch = True
        for patches, masks, _, _, _ in test_loader:
            patches = patches.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(patches)
            probs = torch.sigmoid(logits)

            if first_batch:
                pred_ratio = ((probs > threshold).float().sum().item() / probs.numel()) * 100
                gt_ratio = ((masks == 1).float().sum().item() / masks.numel()) * 100

                logging.info(f"Predicted positives ratio = " f"{pred_ratio:.4f}%")
                logging.info(f"Ground-truth positives ratio = " f"{gt_ratio:.4f}%")

                first_batch = False

            preds_bin = (probs > threshold).float()

            tp += ((preds_bin == 1) & (masks == 1)).sum().item()
            fp += ((preds_bin == 1) & (masks == 0)).sum().item()
            fn += ((preds_bin == 0) & (masks == 1)).sum().item()
            tn += ((preds_bin == 0) & (masks == 0)).sum().item()

    logging.info(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    # FINAL METRICS
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = (2 * prec * rec) / (prec + rec + 1e-6)
    iou  = tp / (tp + fp + fn + 1e-6)

    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall   : {rec:.4f}")
    logging.info(f"F1-score : {f1:.4f}")
    logging.info(f"IoU      : {iou:.4f}")

