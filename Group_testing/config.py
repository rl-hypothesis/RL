import os
from pathlib import Path

THRESHOLD_NORMAL_DIST = 0.05
THRESHOLD_INDEPENDENT_TEST = 0.05

ROOT_DIR = Path(__file__).parent


RESULTS_OUTPUT_PATH = os.path.join(ROOT_DIR, "experiments/results")
TIME_OUTPUT_PATH = os.path.join(ROOT_DIR, "experiments/time")
SCALABILITY_RESULTS = os.path.join(ROOT_DIR, "experiments/scalability")
INTERMEDIATE_RESULTS_FOLDER = os.path.join(ROOT_DIR, "experiments/intermediate")
