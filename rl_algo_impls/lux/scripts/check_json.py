import json
import os

for dirpath, _, filenames in os.walk("data/lux/lux-Deimos"):
    for fname in filenames:
        with open(os.path.join(dirpath, fname)) as f:
            json.load(f)
