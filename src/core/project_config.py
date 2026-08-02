import os

# Project root (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory containing input NetCDF files; override with environment variable DATA_ROOT
DATA_ROOT = os.getenv("DATA_ROOT", "/mnt/data/imerg_correction_hunan")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = os.path.join(MODELS_DIR, "summer")
SPRING_MODELS_DIR = os.path.join(MODELS_DIR, "spring")
SUMMER_RESULTS_DIR = SUMMER_MODELS_DIR
SPRING_RESULTS_DIR = SPRING_MODELS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUMMER_MODELS_DIR, exist_ok=True)
os.makedirs(SPRING_MODELS_DIR, exist_ok=True)
os.makedirs(SUMMER_RESULTS_DIR, exist_ok=True)
os.makedirs(SPRING_RESULTS_DIR, exist_ok=True)

# NOTE: no leading "/" on path segments — ``join(DATA_ROOT, "/hunan/...")`` would ignore DATA_ROOT on POSIX.
IMERG_DIR = os.path.join(DATA_ROOT, "hunan", "imerg_hunan.nc")
OBS_FILE = os.path.join(DATA_ROOT, "cn051.nc")
ERA5_FILE = os.path.join(DATA_ROOT, "hunan", "era5_hunan.nc")
DEM_FILE = os.path.join(DATA_ROOT, "hunan", "dem_hunan_025.nc")

# Training / reproducibility
RANDOM_STATE = 42
MAX_SAMPLES_PER_TIME = 2000
N_ESTIMATORS = 500
TRAIN_SUMMER_MODEL = True
TRAIN_SPRING_MODEL = True

# Train / test years and summer season (months)
TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
TEST_YEARS  = [2021, 2022]
SPRING_MONTHS = [3, 4, 5]
SUMMER_MONTHS = [6, 7, 8]

# Variable name candidates when opening heterogeneous NetCDF sources
IMERG_VAR_CANDIDATES = ["precipitationCal", "precipitation", "precip", "IMERG"]
OBS_VAR_CANDIDATES = ["pre", "precip", "precipitation", "CN05.1"]
U10_VAR_CANDIDATES = ["u10"]
V10_VAR_CANDIDATES = ["v10"]
TCWV_VAR_CANDIDATES = ["tcwv"]
DEM_VAR_CANDIDATES = ["dem", "elevation", "DEM", "height", "__xarray_dataarray_variable__"]

# Saved models and gridded intermediates for plotting
# Seasonal RF models
RF_SUMMER_MODEL_PATH = os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")
RF_SPRING_MODEL_PATH = os.path.join(SPRING_MODELS_DIR, "rf_model_spring.joblib")
# Backward-compatible alias used by older scripts (points to summer model)
RF_MODEL_PATH = RF_SUMMER_MODEL_PATH
LR_MODEL_PATH  = os.path.join(SUMMER_MODELS_DIR, "lr_model.joblib")
DATA_NPZ_PATH  = os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz")
DEM_REF_NC     = os.path.join(SUMMER_MODELS_DIR, "dem_reference.nc")
TIME_NC        = os.path.join(SUMMER_MODELS_DIR, "time_index.nc")
OBS_NC         = os.path.join(SUMMER_MODELS_DIR, "obs_for_plot.nc")
IMERG_NC       = os.path.join(SUMMER_MODELS_DIR, "imerg_for_plot.nc")
U10_NC         = os.path.join(SUMMER_MODELS_DIR, "u10_for_plot.nc")
V10_NC         = os.path.join(SUMMER_MODELS_DIR, "v10_for_plot.nc")
TCWV_NC        = os.path.join(SUMMER_MODELS_DIR, "tcwv_for_plot.nc")
SPRING_LR_MODEL_PATH = os.path.join(SPRING_MODELS_DIR, "lr_model.joblib")
SPRING_DATA_NPZ_PATH = os.path.join(SPRING_MODELS_DIR, "train_test_data.npz")
SPRING_DEM_REF_NC = os.path.join(SPRING_MODELS_DIR, "dem_reference.nc")
SPRING_TIME_NC = os.path.join(SPRING_MODELS_DIR, "time_index.nc")
SPRING_OBS_NC = os.path.join(SPRING_MODELS_DIR, "obs_for_plot.nc")
SPRING_IMERG_NC = os.path.join(SPRING_MODELS_DIR, "imerg_for_plot.nc")
SPRING_U10_NC = os.path.join(SPRING_MODELS_DIR, "u10_for_plot.nc")
SPRING_V10_NC = os.path.join(SPRING_MODELS_DIR, "v10_for_plot.nc")
SPRING_TCWV_NC = os.path.join(SPRING_MODELS_DIR, "tcwv_for_plot.nc")

# Target grid over Hunan (EPSG:4326), ~0.25 deg
TARGET_LON_MIN = 108.65
TARGET_LON_MAX = 114.35
TARGET_LAT_MIN = 24.50
TARGET_LAT_MAX = 30.30
TARGET_RES = 0.25

# plot_figures SHAP: CSV directionality uses full X_test; figures subsample for clarity/speed
SHAP_PLOT_SAMPLES = 300

# --- Manuscript palette (SciencePlots ``styles/science.mplstyle`` axes.prop_cycle) -----------------
# Single source for all figures — avoids a separate module that may be missing on remote workspaces.
SP_IMERG = "#0C5DA5"
SP_RF_FULL = "#00B945"
SP_LR_FULL = "#FF9500"
SP_IDENTITY_LINE = "#9E9E9E"
SP_GRID = "#CFCFCF"
SP_OBS = "#1A1A1A"