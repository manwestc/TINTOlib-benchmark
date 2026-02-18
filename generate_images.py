import torch
import os
import json
import math
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from TINTOlib.tinto import TINTO
from TINTOlib.supertml import SuperTML
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.barGraph import BarGraph
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.combination import Combination
from TINTOlib.featureWrap import FeatureWrap
from TINTOlib.bie import BIE 
#from TINTOlib.fotomics import Fotomics
#from TINTOlib.deepInsight import DeepInsight

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_transformer(method, params, X):
    if method == "IGTD":
        # Auto-set IGTD image size
        columns_minus_one = X.shape[1]  # X already without label
        image_size = math.ceil(math.sqrt(columns_minus_one))
        params["scale"] = [image_size, image_size]
        print(f"[IGTD] Auto-calculated image size: {params['scale']}")

    if method == "TINTO":
        return TINTO(**params)
    elif method == "IGTD":
        return IGTD(**params)
    elif method == "REFINED":
        return REFINED(**params)
    elif method == "BarGraph":
        return BarGraph(**params)
    elif method == "DistanceMatrix":
        return DistanceMatrix(**params)
    elif method == "Combination":
        return Combination(**params)
    elif method == "SuperTML":
        return SuperTML(**params)
    elif method == "FeatureWrap":
        return FeatureWrap(**params)
    elif method == "BIE":
        return BIE(**params)
    #elif method == "Fotomics":
    #    return Fotomics(**params)
    #elif method == "DeepInsight":
     #   return DeepInsight(**params)
    else:
        raise ValueError(f"Unknown transformation method: {method}")

import os
import time
import json

def generate_images_from_config(config, X_train, X_val, X_test):
    all_methods = config["parameters"].keys()

    for alias in all_methods:
        print(f"Generating images with config: {alias}\n")

        # Full parameter block for this alias (e.g., TINTO_blur)
        full_params = config["parameters"][alias]

        # Extract actual method name (e.g., "TINTO")
        method = full_params.get("method", alias)

        # Extract method-specific save path
        method_save_path = full_params["save_path"]

        # Clean parameters for transformer (exclude non-model args)
        transformer_params = {
            k: v for k, v in full_params.items()
            if k not in ["save_path", "method"]
        }

        # Instantiate transformer
        transformer = get_transformer(method, transformer_params, X_train)

        # Make save directories
        train_dir = os.path.join(method_save_path, "train")
        val_dir   = os.path.join(method_save_path, "val")
        test_dir  = os.path.join(method_save_path, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir,   exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        # Path to saved hyperparameters/config (NOT fitted state)
        model_path = os.path.join(method_save_path, "model.pkl")

        # === Timing block with separate fit and transforms ===
        timings = {}

        # 1) Fit on train
        start_fit = time.time()
        transformer.fit(X_train)
        timings["fit_train"] = time.time() - start_fit

        # Save (updated) hyperparameters/config for reuse
        try:
            transformer.saveHyperparameters(model_path)
            print(f"[TINTOlib] Saved hyperparameters to {model_path}")
        except Exception as e:
            print(f"[TINTOlib] Warning: failed to save hyperparameters: {e}")

        # 2) Transform train
        start_tr_train = time.time()
        transformer.transform(X_train, train_dir)
        timings["transform_train"] = time.time() - start_tr_train

        # 3) Transform val
        start_tr_val = time.time()
        transformer.transform(X_val, val_dir)
        timings["transform_val"] = time.time() - start_tr_val

        # 4) Transform test
        start_tr_test = time.time()
        transformer.transform(X_test, test_dir)
        timings["transform_test"] = time.time() - start_tr_test

        # Total time (fit + all transforms)
        timings["total_time"] = (
            timings["fit_train"]
            + timings["transform_train"]
            + timings["transform_val"]
            + timings["transform_test"]
        )

        # Save timings
        timing_file = os.path.join(method_save_path, "generation_times.json")
        with open(timing_file, "w") as f:
            json.dump(timings, f, indent=4)

        # Optional: print timings
        print(f"Time summary for {alias}:")
        for k, v in timings.items():
            print(f"  {k}: {v:.2f} seconds")
        print()




