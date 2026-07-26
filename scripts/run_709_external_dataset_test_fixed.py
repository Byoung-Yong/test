from __future__ import annotations

import numpy as np
import pandas as pd

import run_709_external_dataset_test as validation

_original_choose_model = validation.choose_model


def choose_model_with_normalized_groups(X, y, groups):
    normalized = pd.Series(groups, dtype="object").fillna("__MISSING_SOURCE__").astype(str).to_numpy()
    return _original_choose_model(X, y, normalized)


validation.choose_model = choose_model_with_normalized_groups

if __name__ == "__main__":
    validation.main()
