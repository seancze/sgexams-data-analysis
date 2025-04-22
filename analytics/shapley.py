import matplotlib.pyplot as plt
import joblib
import shap

import sys
from pathlib import Path

# Get the current script's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from utils.format import get_normalised_df


def calculate_shap_values(
    run_dir,
    data_path,
    feature_set,
    model_path,
):
    """Load data and model, preprocess the features, compute SHAP values,
    and display a summary plot."""
    # Load the dataset
    X, _ = get_normalised_df(
        run_dir,
        data_path,
        feature_set,
    )

    # Load the model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")

    # Create a SHAP explainer.
    # For many scikit-learn models, the unified explainer works well.
    try:
        explainer = shap.Explainer(
            model,
            X,
            feature_names=feature_set,
        )
    except Exception as e:
        print(
            "Direct SHAP explainer failed, falling back to KernelExplainer. Error:", e
        )
        # Using a subset of the data as background for KernelExplainer
        background = X.sample(n=min(100, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict, background)

    # Compute SHAP values for the entire validation set
    print("Calculating SHAP values...")
    shap_values = explainer(X)

    # Increase the figure size to accommodate more features
    plt.figure(figsize=(20, 12))

    # Create a summary plot of SHAP values, displaying all features.
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_set,
        max_display=len(feature_set),  # Display all features from feature_set
        show=False,
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/shap-summary.png", bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(20, 12))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_set,
        plot_type="bar",
        max_display=len(feature_set),
        show=False,
    )
    plt.title("SHAP Bar Summary")
    plt.xlabel("Mean(|SHAP value|)")
    ax = plt.gca()
    # Loop through each bar and attach a text label
    for bar in ax.patches:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 1e-8,  # small offset so text isn’t on top of the bar
            y,
            f"{width:.3f}",  # format as you like
            va="center",
            ha="left",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(f"{run_dir}/shap-bar.png", bbox_inches="tight", pad_inches=0)
    plt.close("all")

    print("SHAP values calculated and saved.")

    return shap_values
