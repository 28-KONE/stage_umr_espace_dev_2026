from pystac_client import Client
import planetary_computer
import requests
from pathlib import Path
import os
import time
import geopandas as gpd
from shapely.geometry import box

# CONFIG
WORK = os.environ["WORK"]
SCRATCH = os.environ["SCRATCH"]

OUTPUT_DIR = Path(f"{SCRATCH}/Mangrove_CROMA_Project/data/raw/Sentinel-1_amapa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dates = "2023-01-01/2023-12-31"

# ZONES AMAPA
zones_wgs84 = gpd.GeoDataFrame(
    {
        "zone": [
            "amapa_fg_zone_1",
            "amapa_fg_zone_2",
            "amapa_fg_zone_3",
            "amapa_fg_zone_4",
            "amapa_fg_zone_5",
            "amapa_fg_zone_6",
        ]
    },
    geometry=[
        box(-51.767578, 3.876000, -49.812012, 4.516190),
        box(-51.767578, 3.235000, -49.812012, 3.876000),
        box(-51.767578, 2.594000, -49.812012, 3.235000),
        box(-51.767578, 1.953000, -49.812012, 2.594000),
        box(-51.767578, 1.312000, -49.812012, 1.953000),
        box(-51.767578, 0.681136, -49.812012, 1.312000),
    ],
    crs="EPSG:4326"
)

# STAC SEARCH
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

print("\n▶ Searching Sentinel-1 in Amapa per zone...\n")

all_items = []

for _, row in zones_wgs84.iterrows():

    print(f"Zone: {row.zone}")

    search = catalog.search(
        collections=["sentinel-1-grd"],
        intersects=row.geometry.__geo_interface__,
        datetime=dates,)

    items = list(search.items())

    print(f"   → Found {len(items)} scenes")

    all_items.extend(items)


# REMOVE DUPLICATES
unique_items = list({item.id: item for item in all_items}.values())

print(f"\n Total unique scenes: {len(unique_items)}\n")


# MODE SLURM ARRAY
if "SLURM_ARRAY_TASK_ID" in os.environ:

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if task_id >= len(unique_items):
        print(
            f"⚠ Task {task_id} out of range "
            f"({len(unique_items)} scenes)"
        )
        exit()

    unique_items = [unique_items[task_id]]

    print(
        f" Array mode: "
        f"task {task_id} -> "
        f"{unique_items[0].id}"
    )

# DOWNLOAD
for i, item in enumerate(unique_items):

    try:

        assets = item.assets

        scene_dir = OUTPUT_DIR / item.id
        scene_dir.mkdir(exist_ok=True)

        vv_path = scene_dir / "VV.tif"
        vh_path = scene_dir / "VH.tif"

        if vv_path.exists() and vh_path.exists():

            print(
                f"[{i+1}/{len(unique_items)}] "
                f"⏭ Already downloaded"
            )

            continue

        print(
            f"[{i+1}/{len(unique_items)}] "
            f"Processing {item.id}"
        )

        # VV
        if "vv" in assets and not vv_path.exists():

            print(" Downloading VV")
            signed_url = planetary_computer.sign(assets["vv"].href)
            r = requests.get(signed_url, stream=True, timeout=300,)

            r.raise_for_status()

            with open(vv_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            print(" VV Done")

        # VH
        if "vh" in assets and not vh_path.exists():

            print(" Downloading VH")
            signed_url = planetary_computer.sign(assets["vh"].href)
            r = requests.get(signed_url, stream=True, timeout=300,)

            r.raise_for_status()

            with open(vh_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            print(" VH Done")

        time.sleep(1)

    except Exception as e:
        print(f"❌ Error: {item.id} -> {e}")

print("\n Download finished!\n")
