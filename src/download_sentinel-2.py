import os
import shutil
import zipfile
import rasterio
import datetime
import numpy as np
from pathlib import Path
from shapely.geometry import box, Polygon
from eodag import EODataAccessGateway, setup_logging

# =====================================================
# 1. CONFIGURATION GLOBALE
# =====================================================

YEAR = 2023
PRODUCT_TYPE = "S2MSI2A"  

# AUTHENTIFICATION
os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__USERNAME"] = "damba.kone@umontpellier.fr"
os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__PASSWORD"] = "0767991488Dk@"

# Fenêtres mensuelles
MONTHS = [
    ("2023-01-01", "2023-01-31"),
    ("2023-02-01", "2023-02-28"),
    ("2023-03-01", "2023-03-31"),
    ("2023-04-01", "2023-04-30"),
    ("2023-05-01", "2023-05-31"),
    ("2023-06-01", "2023-06-30"),
    ("2023-07-01", "2023-07-31"),
    ("2023-08-01", "2023-08-31"),
    ("2023-09-01", "2023-09-30"),
    ("2023-10-01", "2023-10-31"),
    ("2023-11-01", "2023-11-30"),
    ("2023-12-01", "2023-12-31"),   
]

#MONTHS = [
#    ("2023-06-01", "2023-06-20"),
#]

# BBOX côtières 
COASTAL_BBOXES = {
    "zone_1": box(-54.0915, 3.9623, -53.6670, 5.8477),
    "zone_2": box(-53.6670, 3.9623, -53.2426, 5.8477),
    "zone_3": box(-53.2426, 3.9623, -52.8182, 5.8477),
    "zone_4": box(-52.8182, 3.9623, -52.3937, 5.8477),
    "zone_5": box(-52.3937, 3.9623, -51.9693, 5.8477),
    "zone_6": box(-51.9693, 3.9623, -51.5448, 5.8477),
}

#COASTAL_BBOXES = {
#    "zone_2": box(-53.6670, 3.9623, -53.2426, 5.8477),
#    "zone_5": box(-52.3937, 3.9623, -51.9693, 5.8477),
#}


BASE_DIR = Path("data/raw/Sentinel-2")
ZIP_DIR = BASE_DIR / "zip"
SAFE_DIR = BASE_DIR / "SAFE"

ZIP_DIR.mkdir(parents=True, exist_ok=True)
SAFE_DIR.mkdir(parents=True, exist_ok=True)

# Filtrage nuages large (le vrai masque sera fait après avec la SCL)
MAX_CLOUD_COVER = 90


# =====================================================
# 2. INITIALISATION EODAG
# =====================================================

def init_dag():
    dag = EODataAccessGateway()
    dag.set_preferred_provider("cop_dataspace")
    return dag


# =====================================================
# 3. TÉLÉCHARGEMENT SENTINEL‑2
# =====================================================

def download_s2(dag, roi, start_date, end_date):
    results = dag.search(
        collection="SENTINEL-2",
        productType=PRODUCT_TYPE,
        geom=roi,
        provider="cop_dataspace",
        start=start_date,
        end=end_date,
        cloudCover=MAX_CLOUD_COVER
    )

    print(f"{len(results)} produits trouvés entre {start_date} et {end_date}")

    downloaded = []

    for product in results:
        try:
            path = dag.download(
                product,
                outputs_prefix=str(ZIP_DIR),
                extract=False
            )
            downloaded.append(path)
        except Exception as e:
            print("Erreur téléchargement :", e)

    return downloaded


# =====================================================
# 4. EXTRACTION
# =====================================================

def extract_archives(zip_paths):
    for zip_path in zip_paths:
        zip_path = Path(zip_path)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(SAFE_DIR)
        except Exception as e:
            print("Erreur extraction :", e)


# =====================================================
# 5. PIPELINE PRINCIPAL
# =====================================================

def main():
    setup_logging(verbose=2)
    dag = init_dag()

    for zone_name, roi in COASTAL_BBOXES.items():
        print(f"\n=== Zone : {zone_name} ===")

        for start_date, end_date in MONTHS:
            print(f"\nTéléchargement S2 {start_date} → {end_date}")

            zip_files = download_s2(
                dag, roi, start_date, end_date
            )
            extract_archives(zip_files)

    print("\nTéléchargement Sentinel‑2 terminé !")


if __name__ == "__main__":
    main()