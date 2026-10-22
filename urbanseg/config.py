"""Central configuration for the Hyderabad segmentation system."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")

DATA_DIR = ROOT / "data"
OSM_DIR = DATA_DIR / "osm"
TILES_DIR = DATA_DIR / "tiles"
MASKS_DIR = DATA_DIR / "masks"
OPEN_BUILDINGS_DIR = DATA_DIR / "open_buildings"
PREVIEW_DIR = DATA_DIR / "preview"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# --- Models ---
# Trained checkpoints in MODELS_DIR, by id. All stay available: pick one with
# URBANSEG_MODEL (an id, or a checkpoint path for local use), the "model"
# field of an API job / the web app, or scripts/segment.py --model.
MODELS = {
    "v1": {"file": "unet_r34.pt", "label": "v1 (August 2026)",
           "about": "trained on the v1 labels: footprints as published, widened by 1.2 m"},
    "v2": {"file": "unet_r34_v2.pt", "label": "v2 (October 2026)",
           "about": "trained on roof-aligned v2 labels, with 16 more areas"},
}
DEFAULT_MODEL = "v2"

# --- Imagery ---
ZOOM = 18          # ~0.57 m per map px at Hyderabad latitude
SCALE = 2          # image px = 2x map px  -> ~0.285 m per image px
TILE_MAP_PX = 640  # Static Maps request size (map px)
CROP_BOTTOM_MAP_PX = 40   # strip Google watermark row
USABLE_X = TILE_MAP_PX                       # 640 map px
USABLE_Y = TILE_MAP_PX - CROP_BOTTOM_MAP_PX  # 600 map px

# --- Classes ---
# 0 other (roads / rail / water), 1 residential building,
# 2 commercial building (incl. office/retail/industrial/institutional),
# 3 free land (any unbuilt ground: bare earth, scrub, fields, vegetation)
CLASS_NAMES = ["other", "residential", "commercial", "free_land"]
N_CLASSES = 4
CLASS_COLORS = {  # RGBA overlay colors
    0: (120, 120, 120),
    1: (255, 170, 0),
    2: (220, 40, 40),
    3: (80, 200, 120),
}

# Greater Hyderabad bounding box used to filter Open Buildings
HYD_BBOX = (78.10, 17.10, 78.80, 17.70)  # west, south, east, north

# --- Training areas ---
# (name, center_lat, center_lng, grid_nx, grid_ny, split)
# grid tiles are USABLE_X x USABLE_Y map px each, laid out around the center.
AREAS = [
    # residential-heavy
    ("kphb",           17.4931, 78.3996, 3, 3, "train"),
    ("nizampet",       17.5169, 78.3907, 3, 3, "train"),
    ("miyapur",        17.4948, 78.3562, 3, 3, "train"),
    ("uppal",          17.4056, 78.5591, 3, 3, "train"),
    ("dilsukhnagar",   17.3688, 78.5247, 3, 3, "train"),
    ("vanasthalipuram",17.3436, 78.5548, 3, 3, "train"),
    ("alwal",          17.5047, 78.5039, 3, 3, "train"),
    ("manikonda",      17.4024, 78.3868, 3, 3, "train"),
    ("oldcity",        17.3616, 78.4747, 3, 3, "train"),
    ("banjara",        17.4108, 78.4294, 3, 3, "train"),
    # commercial / industrial
    ("hitech",         17.4435, 78.3772, 3, 3, "train"),
    ("gachibowli_fd",  17.4239, 78.3428, 3, 3, "train"),
    ("ameerpet",       17.4374, 78.4482, 3, 3, "train"),
    ("abids",          17.3906, 78.4772, 3, 3, "train"),
    ("sdroad",         17.4399, 78.4983, 3, 3, "train"),
    ("jeedimetla",     17.5211, 78.4456, 3, 3, "train"),
    ("balanagar",      17.4753, 78.4409, 3, 3, "train"),
    # free land / outskirts
    ("kokapet",        17.3910, 78.3316, 3, 3, "train"),
    ("adibatla",       17.2403, 78.5533, 3, 3, "train"),
    ("shamshabad",     17.2570, 78.4055, 3, 3, "train"),
    ("ghatkesar",      17.4489, 78.6839, 3, 3, "train"),
    # validation (used for early stopping / model choice)
    ("lbnagar",        17.3498, 78.5513, 3, 3, "val"),
    ("kompally",       17.5453, 78.4854, 3, 3, "val"),
    # held-out test (never seen in training; accuracy is reported here)
    ("madhapur",       17.4482, 78.3915, 3, 3, "test"),
    ("begumpet",       17.4440, 78.4661, 3, 3, "test"),
    ("kondapur",       17.4647, 78.3639, 3, 3, "test"),
]

# --- Version-2 areas (October 2026) ---
# data/ is the frozen v1 dataset. New areas, and every label version after
# v1, live in data_v2/ (tiles/, osm/, masks_<version>/), never in data/.
# The v1 areas are almost all dense; these cover the textures they miss.
# (name, center_lat, center_lng, grid_nx, grid_ny, split, texture); "test2"
# areas are held out, one per new texture, and lie at least 2 km from every
# training area (Hussain Sagar and Jubilee Hills are near the v1 test areas,
# which are held out too).
DATA_V2_DIR = ROOT / "data_v2"
# hand-reviewed gold samples (urbanseg.gold), written by the web app's editor
GOLD_DIR = Path(os.environ.get("URBANSEG_GOLD_DIR") or DATA_V2_DIR / "gold")
AREAS_V2 = [
    ("himayat_sagar",   17.3150, 78.3700, 3, 3, "train", "water"),
    ("shamirpet",       17.5900, 78.5650, 3, 3, "train", "water"),
    ("tukkuguda",       17.2050, 78.4800, 3, 3, "train", "plots"),
    ("ibrahimpatnam",   17.1900, 78.6500, 3, 3, "train", "plots"),
    ("hayathnagar",     17.3280, 78.6000, 3, 3, "train", "plots"),
    ("kollur",          17.4600, 78.2560, 3, 3, "train", "plots"),
    ("bachupally",      17.5450, 78.3820, 3, 3, "train", "plots"),
    ("village_east",    17.5486, 78.7128, 3, 3, "train", "plots"),
    ("industry_west",   17.3266, 78.1360, 3, 3, "train", "industry"),
    ("industry_north",  17.6781, 78.5686, 3, 3, "train", "industry"),
    ("patancheru",      17.5300, 78.2700, 3, 3, "train", "industry"),
    ("trimulgherry",    17.4760, 78.5150, 3, 3, "train", "cantonment"),
    ("sainikpuri",      17.4930, 78.5480, 3, 3, "train", "cantonment"),
    ("tolichowki",      17.3930, 78.4110, 3, 3, "train", "old_informal"),
    ("tellapur",        17.4560, 78.2880, 3, 3, "train", "highrise"),
    ("narsingi",        17.3880, 78.3550, 3, 3, "train", "highrise"),
    # held out
    ("hussain_sagar",   17.4240, 78.4740, 3, 3, "test2", "water"),
    ("jubilee_hills",   17.4310, 78.4070, 3, 3, "test2", "villas"),
    ("chandrayangutta", 17.3110, 78.4830, 3, 3, "test2", "old_informal"),
    ("medchal",         17.6300, 78.4800, 3, 3, "test2", "plots"),
    ("cherlapally",     17.4620, 78.6000, 3, 3, "test2", "industry"),
]

# --- OSM classification vocabularies ---
RES_BUILDING_TAGS = {
    "residential", "apartments", "house", "detached", "semidetached_house",
    "terrace", "hut", "dormitory", "bungalow", "ger", "cabin",
}
COM_BUILDING_TAGS = {
    "commercial", "retail", "office", "industrial", "warehouse", "supermarket",
    "mall", "hotel", "hospital", "school", "college", "university", "kiosk",
    "government", "civic", "public", "temple", "mosque", "church", "religious",
    "train_station", "transportation", "sports_hall", "stadium", "cinema",
    "manufacture", "factory", "service", "shed",
}
COM_LANDUSE = {"commercial", "retail", "industrial", "institutional", "education", "religious"}
RES_LANDUSE = {"residential"}

# amenity POI values that imply a commercial/institutional building
AMENITY_COM = {
    "restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "bank",
    "pharmacy", "hospital", "clinic", "doctors", "dentist", "veterinary",
    "school", "college", "university", "kindergarten", "driving_school",
    "fuel", "marketplace", "cinema", "theatre", "post_office", "police",
    "townhall", "community_centre", "library", "place_of_worship",
    "events_venue", "conference_centre", "internet_cafe", "car_rental",
    "car_wash", "money_transfer", "bureau_de_change",
}

# buildings that should be painted as class 0 (transport infrastructure)
OTHER_BUILDING_TAGS = {"train_station", "transportation"}

# per-area training sampling weights from the label-quality audit
# (0 = excluded; val/test areas ignore this)
AREA_WEIGHTS = {
    "nizampet": 0.0,
    "uppal": 1.0, "jeedimetla": 1.0,
    "hitech": 0.9, "gachibowli_fd": 0.9, "ameerpet": 0.9, "shamshabad": 0.9,
    "kphb": 0.8, "dilsukhnagar": 0.8, "oldcity": 0.8, "banjara": 0.8,
    "sdroad": 0.8, "alwal": 0.8, "miyapur": 0.8, "adibatla": 0.8,
    "vanasthalipuram": 0.7, "abids": 0.7, "balanagar": 0.7, "kokapet": 0.7,
    "ghatkesar": 0.7, "manikonda": 0.6,
}

# road buffer full widths in meters by highway class (buffer uses half)
ROAD_WIDTHS = {
    "motorway": 20, "trunk": 18, "primary": 16, "secondary": 12,
    "tertiary": 9, "unclassified": 6, "residential": 6, "service": 4,
    "living_street": 5, "motorway_link": 10, "trunk_link": 10,
    "primary_link": 10, "secondary_link": 8, "tertiary_link": 6,
}
RAIL_WIDTH = 8
