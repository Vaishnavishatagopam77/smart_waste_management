from pathlib import Path  # To work with file and folder paths
import sys  # To access system-specific parameters and functions

# Get the full path of the current file (settings.py)
file_path = Path(__file__).resolve()

# Get the directory path of the current file
root_path = file_path.parent

# Add the root path to system path if it's not already added (so we can import files easily)
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the path relative to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# ==== ML MODEL CONFIGURATION ====

# Folder where the model weights are stored
MODEL_DIR = ROOT / 'weights'

# Path to the trained YOLO model file
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# ==== WEBCAM CONFIGURATION ====

# Use default webcam (0 is usually the built-in webcam)
WEBCAM_PATH = 0

# ==== TYPES OF WASTE EXAMPLES (for understanding waste categories) ====

# 'R' = Recyclable
# 'N' = Non-Recyclable
# 'H' = Hazardous

# ==== CLASSIFYING WASTE ITEMS ====

# List of recyclable waste items
RECYCLABLE = [
    'cardboard_box',
    'can',
    'plastic_bottle_cap',
    'plastic_bottle',
    'reuseable_paper'
]

# List of non-recyclable waste items
NON_RECYCLABLE = [
    'plastic_bag',
    'scrap_paper',
    'stick',
    'plastic_cup',
    'snack_bag',
    'plastic_box',
    'straw',
    'plastic_cup_lid',
    'scrap_plastic',
    'cardboard_bowl',
    'plastic_cultery'
]

# List of hazardous waste items
HAZARDOUS = [
    'battery',
    'chemical_spray_can',
    'chemical_plastic_bottle',
    'chemical_plastic_gallon',
    'light_bulb',
    'paint_bucket'
]
