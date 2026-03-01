import os
import yaml

# === Project and config file paths ===
PROJECT_ROOT = os.getenv(
    'PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
)
CONFIG_PATH = os.getenv(
    'CONFIG_PATH',
    os.path.join(PROJECT_ROOT, 'config.yaml')
)

# === Load YAML configuration ===
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# === Dataset root ===
DATASET_ROOT = os.path.join(PROJECT_ROOT, cfg['dataset']['root'])

# === Train split ===
TRAIN_IMAGES_DIR = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['train']['images']
)
TRAIN_COCO_JSON = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['train']['coco_json']
)

# === Validation split ===
VALID_IMAGES_DIR = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['valid']['images']
)
VALID_COCO_JSON = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['valid']['coco_json']
)

# === Test split ===
TEST_IMAGES_DIR = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['test']['images']
)
TEST_COCO_JSON = os.path.join(
    DATASET_ROOT,
    cfg['dataset']['test']['coco_json']
)

METRIC_THRESHOLD = cfg['metrics']['threshold']

ENCODER = cfg.get('encoder', 'resnet34')  # Default to resnet34 if not specified

# === Other settings placeholders ===
# CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
# LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
