import numpy as np
import os
import joblib
import xgboost as xgb
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Get the current script's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from utils.constants import (
    SCALER_FILENAME,
    ALL_FEATURES,
    OUTPUT_CSV_FILEPATH,
    OUTPUT_EXCEL_FILEPATH,
    TRAIN_FILEPATH,
    VALIDATION_FILEPATH,
    TEST_FILEPATH,
)
from utils.format import get_numerical_columns


def create_run_directory(run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/{run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_train_test_val_split(df):
    X = df[ALL_FEATURES].copy()
    y = df["is_viral"].copy()
    random_state = 42
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(X)),
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,  # preserve the class distribution in both splits
    )

    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,  # preserve the class distribution in both splits
    )

    # Create feature subsets using indices
    X_train = X.iloc[X_train_idx]
    X_val = X.iloc[X_val_idx]
    X_test = X.iloc[X_test_idx]

    # save training, validation, test into separate .csv files
    df_train = df.iloc[X_train_idx]
    df_val = df.iloc[X_val_idx]
    df_test = df.iloc[X_test_idx]

    df_train.to_csv(TRAIN_FILEPATH, index=False)
    df_val.to_csv(VALIDATION_FILEPATH, index=False)
    df_test.to_csv(TEST_FILEPATH, index=False)

    print(f"TRAINING SET")
    print(y_train.value_counts() / len(y_train))
    print(f"VALIDATION SET")
    print(y_val.value_counts() / len(y_val))
    print(f"TEST SET")
    print(y_test.value_counts() / len(y_test))

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test


# Define the styling function for TRUE/FALSE values
def _highlight_true(val):
    GREEN_STYLE = "background-color: #06b050"
    if isinstance(val, bool) and val:
        return GREEN_STYLE
    if isinstance(val, str) and val.upper() == "TRUE":
        return GREEN_STYLE
    return ""


def save_results(df):
    # Reorder columns to match desired format
    column_order = ALL_FEATURES + [
        "num_features",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "best_model",
        # add test results to the end of the csv
        # test results will only have a value if GET_TEST_PERFORMANCE is True
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]

    # reorder the columns to match the desired format
    # if columns are missing, fill with empty strings
    df = df.reindex(columns=column_order, fill_value="")

    # Create a Styler object
    styler = df.style

    # Apply the styling to feature columns only
    for col in ALL_FEATURES:
        styler = styler.applymap(_highlight_true, subset=[col])

    # Save results with styling
    styler.to_excel(OUTPUT_EXCEL_FILEPATH, index=False)
    # Also save as CSV which will not include the styling
    df.to_csv(OUTPUT_CSV_FILEPATH, index=False)


def train_and_evaluate(
    run_dir,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    feature_set,
    df_val,  # used solely to save best validation model results
    random_state=42,
):
    # Prepare features for each dataset
    X_train_subset = X_train[feature_set].copy()
    X_val_subset = X_val[feature_set].copy()
    X_test_subset = X_test[feature_set].copy()

    # Scale numerical features
    numerical_columns = get_numerical_columns(feature_set)
    if numerical_columns:
        scaler = StandardScaler()
        # Fit on training data
        X_train_subset[numerical_columns] = scaler.fit_transform(
            X_train_subset[numerical_columns]
        )
        # Apply same transformation to validation and test
        X_val_subset[numerical_columns] = scaler.transform(
            X_val_subset[numerical_columns]
        )
        X_test_subset[numerical_columns] = scaler.transform(
            X_test_subset[numerical_columns]
        )

        # save scaler and label encoder to .pkl files
        print(f"Saving scaler encoders...")
        with open(f"{run_dir}/{SCALER_FILENAME}", "wb") as f:
            joblib.dump(scaler, f)
        print(f"Successfully saved scaler encoder")

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(random_state=random_state),
    }

    best_val_f1_for_experiment = 0
    best_model_for_experiment = None
    best_model_name_for_experiment = None
    best_val_pred_for_experiment = None

    for name, model in models.items():
        # train model on training set
        model.fit(X_train_subset, y_train)
        # predict on validation set
        y_val_pred = model.predict(X_val_subset)
        val_f1 = f1_score(y_val, y_val_pred, average="macro")

        if val_f1 > best_val_f1_for_experiment:
            best_val_f1_for_experiment = val_f1
            best_model_for_experiment = model
            best_model_name_for_experiment = name
            best_val_pred_for_experiment = y_val_pred

    # keep track of results for the best model
    val_metrics = {
        "accuracy": accuracy_score(y_val, best_val_pred_for_experiment) * 100,
        "precision": precision_score(
            y_val, best_val_pred_for_experiment, average="macro"
        )
        * 100,
        "recall": recall_score(y_val, best_val_pred_for_experiment, average="macro")
        * 100,
        "f1_score": f1_score(y_val, best_val_pred_for_experiment, average="macro")
        * 100,
        "best_model": best_model_name_for_experiment,
    }

    # used for debugging purposes
    validation_results = df_val.copy()
    validation_results["actual_label"] = y_val
    validation_results["predicted_label"] = best_val_pred_for_experiment.astype(bool)

    return (
        val_metrics,
        (best_model_for_experiment, X_test_subset),
        validation_results,
    )
