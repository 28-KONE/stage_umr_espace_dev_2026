import geopandas as gpd
from shapely.geometry import box

# ============================
# PARAMÈTRES
# ============================

# Chemin vers le fichier vecteur mangrove 2023
MANGROVE_FILE = "C:/Users/Kone/Desktop/STAGE UMR ESPACE DEV/VSCode/vecteurs_mangroves_2023"
BUFFER_DISTANCE = 10_000  # 10 km en mètres
N_BBOX = 6  # nombre de bandes côtières

# =====================================================
# LECTURE DU VECTEUR MANGROVE
# =====================================================

gdf = gpd.read_file(MANGROVE_FILE)

# =====================================================
# 1. PROJETER EN CRS MÉTRIQUE (obligatoire pour buffer)
# =====================================================

# Guyane → UTM zone 21N
gdf = gdf.to_crs(epsg=32621)

# =====================================================
# 2. BUFFER DE 10 KM
# =====================================================

gdf_buffer = gdf.buffer(BUFFER_DISTANCE)
gdf_buffer = gdf_buffer.union_all()  # dissoudre

# =====================================================
# 3. RETOUR EN WGS84
# =====================================================

gdf_buffer = gpd.GeoSeries([gdf_buffer], crs=32621).to_crs(epsg=4326)

# =====================================================
# 4. EMPRISE GLOBALE DU BUFFER
# =====================================================

minx, miny, maxx, maxy = gdf_buffer.total_bounds

# =====================================================
# 5DÉCOUPE AUTOMATIQUE EN BANDES
# =====================================================

width = (maxx - minx) / N_BBOX

print("\nCOASTAL_BBOXES = {")

for i in range(N_BBOX):
    x1 = minx + i * width
    x2 = minx + (i + 1) * width
    print(
        f'    "zone_{i+1}": box({x1:.4f}, {miny:.4f}, {x2:.4f}, {maxy:.4f}),'
    )

print("}")