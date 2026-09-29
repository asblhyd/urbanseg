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
