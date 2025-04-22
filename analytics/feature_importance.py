import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

import sys
from pathlib import Path


# Get the current script's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from utils.format import get_normalised_df


def calculate_feature_importance(
    run_dir,
    data_path,
    feature_set,
    model_path,
):
    """Calculate and visualize permutation feature importance using saved scaler and label encoders."""

    X, y = get_normalised_df(
        run_dir,
        data_path,
        feature_set,
    )

    # Load the model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")

    # Calculate permutation importance
    print("Calculating permutation importance...")
    result = permutation_importance(
        model, X, y, n_repeats=30, random_state=42, n_jobs=-1, scoring="f1_macro"
    )

    # Create results DataFrame
    importance_df = pd.DataFrame(
        {
            "Feature": feature_set,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    ).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    output_filepath = f"{run_dir}/feature_importance.csv"
    importance_df.to_csv(output_filepath, index=False)

    print(f"\nResults saved to {output_filepath}")

    return importance_df
