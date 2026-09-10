
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

OUTPUT_DIR = Path(f"{SCRATCH}/Mangrove_CROMA_Project/data/raw/Sentinel-2_amapa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dates = "2023-01-01/2023-12-31"

zones_wgs84 = gpd.GeoDataFrame({
        "zone": [
            "amapa_fg_zone_1",
            "amapa_fg_zone_2",
            "amapa_fg_zone_3",
            "amapa_fg_zone_4",
            "amapa_fg_zone_5",
            "amapa_fg_zone_6",
        ]},
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

# STAC
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

print("\n▶ Searching Sentinel-2 L2A per zone...\n")

all_items = []

for _, row in zones_wgs84.iterrows():

    print(f"Zone: {row.zone}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=row.geometry.__geo_interface__,
        datetime=dates,
        query={"eo:cloud_cover": {"lt": 80}})

    items = list(search.items())

    print(f"   → Found {len(items)} scenes")

    all_items.extend(items)


# REMOVE DUPLICATES
unique_items = list(
    {item.id: item for item in all_items}.values())

print(f"\n Total unique scenes: {len(unique_items)}\n")


# BANDS
bands = [
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
    "SCL",
]


# DOWNLOAD
for i, item in enumerate(unique_items):
    try:
        assets = item.assets
        scene_dir = OUTPUT_DIR / item.id
        scene_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i+1}/{len(unique_items)}] "  f"Processing {item.id}")

        for band in bands:
            if band not in assets:
                print(f"   ⚠ Missing {band}")
                continue

            out_file = scene_dir / f"{band}.tif"
            if out_file.exists():
                print(f"   ⏭ {band} already exists")
                continue

            print(f" Downloading {band}")

            signed_url = planetary_computer.sign(assets[band].href)
            r = requests.get(signed_url, stream=True, timeout=300)
            r.raise_for_status()

            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f" {band} done")

        time.sleep(1)

    except Exception as e:
        print(f"❌ Error on {item.id} -> {e}")

print("\n Download finished!\n")
