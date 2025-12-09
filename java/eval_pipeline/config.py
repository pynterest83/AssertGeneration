"""
Configuration file - Paths to datasets and settings
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent

# RQ1 paths
RQ1_BASE = BASE_DIR / "RQ1"
RQ1_DATASET = RQ1_BASE / "fine-tuning" / "dataset" / "dataset_codeparrot.pickle"
RQ1_INPUTS = RQ1_BASE / "fine-tuning" / "dataset" / "inputs.csv"
RQ1_META = RQ1_BASE / "fine-tuning" / "dataset" / "meta.csv"

# RQ2 paths
RQ2_BASE = BASE_DIR / "RQ2"
RQ2_INFERENCE_DATA = RQ2_BASE / "inference" / "inference_data"
RQ2_RESULTS = RQ2_BASE / "inference" / "results"

# RQ4 paths
RQ4_BASE = BASE_DIR / "RQ4"
RQ4_PROJECTS = RQ4_BASE / "artifacts_with_es_togll_tests"

# RQ5 paths
RQ5_BASE = BASE_DIR / "RQ5"
RQ5_INPUTS = RQ5_BASE / "TOGLL_prediction" / "evosuite_reaching_tests" / "inputs.csv"
RQ5_META = RQ5_BASE / "TOGLL_prediction" / "evosuite_reaching_tests" / "meta.csv"
RQ5_DATASET = RQ5_BASE / "TOGLL_prediction" / "input_data" / "defects4j_codeparrot.pickle"

# Projects list for RQ2 and RQ4
PROJECTS = [
    "async-http-client",
    "bcel-6.5.0-src",
    "commons-beanutils-1.9.4",
    "commons-collections4-4.4-src",
    "commons-configuration2-2.8.0-src",
    "commons-dbutils-1.7",
    "commons-geometry-1.0-src",
    "commons-imaging-1.0-alpha3-src",
    "commons-jcs3-3.1-src",
    "commons-jexl3-3.2.1-src",
    "commons-lang3-3.12.0-src",
    "commons-net-3.8.0",
    "commons-numbers-1.0-src",
    "commons-pool2-2.11.1-src",
    "commons-rng-1.4-src",
    "commons-validator-1.7",
    "commons-vfs-2.9.0",
    "commons-weaver-2.0-src",
    "http-request",
    "joda-time",
    "JSON-java",
    "jsoup",
    "scribejava",
    "spark",
    "springside4"
]

# Settings
BATCH_SIZE = 32
MAX_LENGTH = 512
NUM_BEAMS = 5

