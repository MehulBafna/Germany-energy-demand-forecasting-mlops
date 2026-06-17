import os
from pathlib import Path

# Change to project root at import time so params.yaml is found by all src modules
os.chdir(Path(__file__).parent.parent)
