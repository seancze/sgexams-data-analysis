import time
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import argparse


from utils.constants import (
    POSITIVE_PERMUTATION_FEATURES,
    INPUT_CSV_FILEPATH,
    OUTPUT_CSV_FILEPATH,
    ALL_FEATURES,
    VALIDATION_FILEPATH,
)
from utils.experiment import (
    save_results,
    get_train_test_val_split,
    create_run_directory,
    train_and_evaluate,
)
from utils.format import format_df
from analytics.feature_importance import calculate_feature_importance
from analytics.shapley import calculate_shap_values


def _get_existing_results():
    # Load existing results if file exists
    results_list = []
    existing_results = None
    best_val_f1 = 0

    try:
        existing_results = pd.read_csv(OUTPUT_CSV_FILEPATH)

        # get the best validation f1-score from existing results
        best_val_f1 = existing_results["f1_score"].max()

        # Get missing columns by comparing with ALL_FEATURES
        missing_columns = [
            col for col in ALL_FEATURES if col not in existing_results.columns
        ]

        # Add missing columns with NaN values if any
        if missing_columns:
            for col in missing_columns:
                print(f"Missing column: {col}")
                existing_results[col] = "FALSE"

        # orient="records" returns a list of dictionaries
        results_list = existing_results.to_dict(orient="records")

    except FileNotFoundError:
        pass

    return results_list, existing_results, best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select which training to run, whether to run all experiments, and whether to evaluate on the test set"
    )
    # mode: best model (0) or all features (1)
    parser.add_argument(
        "--mode",
        choices=["0", "1"],
        default="0",
        help="0: train best model (default), 1: train model with ALL features",
    )
    # run all experiments instead of just one
    parser.add_argument(
        "--all",
        dest="run_all_experiments",
        action="store_true",
        help="If set, all experiments will be performed (default: run only the one selected by --mode). If this flag is set, it overwrites --mode.\n\nWARNING: This might cause a segmentation fault if you have insufficient memory",
    )
    parser.add_argument(
        "--test",
        dest="get_test_performance",
        action="store_true",
        help="If set, evaluate on the test dataset (default: validation only)",
    )
    args = parser.parse_args()
    TRAIN_BEST_MODEL = args.mode == "0"
    TRAIN_ALL_FEATURES = args.mode == "1"
    RUN_ALL_EXPERIMENTS = args.run_all_experiments
    GET_TEST_PERFORMANCE = args.get_test_performance

    # Load the data
    df = pd.read_csv(INPUT_CSV_FILEPATH)

    df = format_df(df)

    features_used = []

    if RUN_ALL_EXPERIMENTS:
        print(f"Running all experiments...")
        for i in range(len(POSITIVE_PERMUTATION_FEATURES)):
            temp = POSITIVE_PERMUTATION_FEATURES[: i + 1]
            features_used.append(temp)
    elif TRAIN_BEST_MODEL:
        # we know that this is the experiment that will train the best model because we have ran all experiments before
        print(f"Running best model experiment...")
        features_used = [POSITIVE_PERMUTATION_FEATURES[:32]]
    else:
        print(f"Running all features experiment...")
        features_used = [ALL_FEATURES]

    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test = (
        get_train_test_val_split(df)
    )

    # Run experiments for each feature set
    results_list, existing_results, best_val_f1 = _get_existing_results()
    best_experiment_data = None
    best_validation_results = None
    best_model_run_dir = None
    best_model_feature_set = None

    for i, feature_set in enumerate(features_used):
        start_time = time.time()
        # Check if experiment already exists in previous results
        if existing_results is not None:
            # Create a mask for current feature set
            feature_mask = pd.Series([True] * len(existing_results))
            for feature in ALL_FEATURES:
                if feature in feature_set:
                    feature_mask &= (
                        existing_results[feature].astype(str).str.upper() == "TRUE"
                    )
                else:
                    feature_mask &= (
                        existing_results[feature].astype(str).str.upper() == "FALSE"
                    )

            # if the experiment exists and has a valid f1 score, skip it
            if (
                feature_mask.any()
                and not existing_results.loc[feature_mask, "f1_score"].isna().all()
            ):
                print(
                    f"[INFO] Experiment SKIPPED as it already exists. To re-run the experiment, remove it from {OUTPUT_CSV_FILEPATH}"
                )
                existing_experiment = existing_results[feature_mask].iloc[0].to_dict()
                continue

        run_dir = create_run_directory(f"{len(feature_set)}_features")
        val_metrics, experiment_data, validation_results = train_and_evaluate(
            run_dir,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            feature_set,
            df_val,
        )
        cur_model, _ = experiment_data
        print(f"model_type: {type(cur_model)}")
        # save current model to a .pkl file
        joblib.dump(cur_model, f"{run_dir}/model.pkl")

        row = {col: "TRUE" if col in feature_set else "FALSE" for col in ALL_FEATURES}
        row["num_features"] = len(feature_set)

        row.update(val_metrics)
        results_list.append(row)

        # track best performing experiment based on validation F1-score
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_experiment_data = experiment_data
            best_validation_results = validation_results
            best_model_run_dir = run_dir
            best_model_feature_set = feature_set
        print(f"Experiment {i+1} completed in {time.time() - start_time:.2f} seconds")

    # save the validation results for the best model for debugging/analysis purposes
    if best_validation_results is not None:
        best_validation_results.to_csv(
            f"{best_model_run_dir}/best_model_validation_predictions.csv", index=False
        )
        model_path = f"{best_model_run_dir}/model.pkl"
        calculate_feature_importance(
            best_model_run_dir, VALIDATION_FILEPATH, best_model_feature_set, model_path
        )
        calculate_shap_values(
            best_model_run_dir, VALIDATION_FILEPATH, best_model_feature_set, model_path
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort results by f1_score in descending order
    results_df = results_df.sort_values("f1_score", ascending=False)

    # get index of row with the best validation f1-score
    best_row_idx = results_df["f1_score"].idxmax()
    best_row = results_df.loc[best_row_idx]
    print("\nBest Feature Set Results:")
    print(f"Features used: {[col for col in ALL_FEATURES if best_row[col] == 'TRUE']}")
    print(f"Number of features used: {best_row['num_features']}")
    print(f"Best model: {best_row['best_model']}")
    print("\nValidation Metrics:")
    print(f"Accuracy: {best_row['accuracy']:.2f}%")
    print(f"Precision: {best_row['precision']:.2f}%")
    print(f"Recall: {best_row['recall']:.2f}%")
    print(f"F1-Score: {best_row['f1_score']:.2f}%")

    # Evaluate best model on test set
    if best_experiment_data is not None and GET_TEST_PERFORMANCE:
        best_model, X_test = best_experiment_data
        # save the best model
        joblib.dump(best_model, f"{run_dir}/best_model.pkl")
        y_test_pred = best_model.predict(X_test)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred) * 100,
            "test_precision": precision_score(y_test, y_test_pred, average="macro")
            * 100,
            "test_recall": recall_score(y_test, y_test_pred, average="macro") * 100,
            "test_f1_score": f1_score(y_test, y_test_pred, average="macro") * 100,
        }

        for metric, value in test_metrics.items():
            results_df.loc[best_row_idx, metric] = value

        print("\nTest Metrics:")
        print(f"Accuracy: {test_metrics['test_accuracy']:.2f}%")
        print(f"Precision: {test_metrics['test_precision']:.2f}%")
        print(f"Recall: {test_metrics['test_recall']:.2f}%")
        print(f"F1-Score: {test_metrics['test_f1_score']:.2f}%")

    # save experiment results to both a .csv and .xlsx file
    save_results(results_df)
