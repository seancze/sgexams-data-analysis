import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Get the current script's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from utils.constants import (
    BOOLEAN_COLUMNS,
    CYCLICAL_COLUMNS,
    SCALER_FILENAME,
    ORDINAL_COLUMNS,
)


def _categorise_time_24h(time_of_day):
    if time_of_day in {23, 0, 1, 4}:
        return "low_upvotes"
    elif time_of_day in {9, 12}:
        return "high_upvotes"
    else:
        return "others"


def _categorise_word_count(word_count):
    if word_count >= 0 and word_count <= 99:
        return 0
    elif word_count >= 100 and word_count <= 199:
        return 1
    elif word_count >= 200 and word_count <= 299:
        return 2
    elif word_count >= 300 and word_count <= 399:
        return 3
    elif word_count >= 400 and word_count <= 499:
        return 4
    elif word_count >= 500 and word_count <= 599:
        return 5
    elif word_count >= 600 and word_count <= 699:
        return 6
    elif word_count >= 700 and word_count <= 799:
        return 7
    elif word_count >= 800 and word_count <= 899:
        return 8
    else:
        return 9


def get_numerical_columns(feature_set):
    """Get numerical columns from the feature set."""
    numerical_columns = []
    # for simple ML classifiers, there should be no need to normalise one-hot encoded features
    # see here: https://stats.stackexchange.com/questions/399430/does-categorical-variable-need-normalization-standardization
    # NOTE: empirical findings show no difference between normalising and not normalising ordinal columns
    for col in feature_set:
        if col in BOOLEAN_COLUMNS or col in CYCLICAL_COLUMNS or col in ORDINAL_COLUMNS:
            continue
        numerical_columns.append(col)
    return numerical_columns


def get_normalised_df(
    run_dir,
    data_path,
    feature_set,
    scaler_filename=SCALER_FILENAME,
):
    # Load the dataset
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    # Prepare validation features
    X = df[feature_set].copy()

    # Get target variable from validation data
    y = df["is_viral"]

    # Transform boolean columns to integers
    for col in BOOLEAN_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Scale numerical features using the existing scaler
    numerical_columns = get_numerical_columns(feature_set)
    if numerical_columns:
        scaler_filepath = f"{run_dir}/{scaler_filename}"
        print(f"Loading scaler from {scaler_filepath}")
        scaler = joblib.load(scaler_filepath)  # Load the pre-fitted scaler
        X[numerical_columns] = scaler.transform(X[numerical_columns])

    return X, y


def format_df(df):
    """
    Process and augment a Reddit posts DataFrame for modeling.

    This function computes word counts for post text, bins word counts and posting hours,
    one-hot encodes categorical features (time of day, theme, emotion, sentiment), applies
    cyclical encoding to the hour of day, normalises new column names to lowercase with
    underscores, and removes posts with no keyword matches.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - selftext : text of the post
        - hour_24h : integer hour of posting (0–23)
        - theme_merged : merged theme label
        - emotion : emotion label
        - sentiment : sentiment label
        - Words_Found : count of keywords found in the post

    Returns
    -------
    pandas.DataFrame
        A new DataFrame including:
        - word_count : number of words in selftext
        - word_count_bins : binned word count via `categorise_word_count`
        - hour_24h_bins : binned posting hour via `categorise_time_24h`
        - one-hot columns for each bin/category in time_of_day_*, theme_*, emotion_*, sentiment_*
        - hour_sin, hour_cos : cyclical encodings of hour_24h
        Rows where `Words_Found == 0` are dropped.

    Side Effects
    ------------
    Prints the number of rows removed due to `Words_Found == 0`.
    """
    # if word_count does not exist
    if "word_count" not in df.columns:
        # get word count for the "selftext" column
        df["word_count"] = df["selftext"].apply(lambda x: len(str(x).split()))

    # Categorise word count into bins
    df["word_count_bins"] = df["word_count"].apply(_categorise_word_count)

    # Categorise time of day into bins
    df["hour_24h_bins"] = df["hour_24h"].apply(_categorise_time_24h)

    # --- One-hot encode the "hour_24h_bins" column ---
    # Create dummy variables for the "hour_24h_bins" column
    hour_dummies = pd.get_dummies(df["hour_24h_bins"], prefix="time_of_day")
    # Concatenate the dummies with the DataFrame and drop the original column
    df = pd.concat([df, hour_dummies], axis=1)

    # --- One-hot encode the "theme" column ---
    # Create dummy variables for the "theme" column
    theme_dummies = pd.get_dummies(df["theme_merged"], prefix="theme")
    # Concatenate the dummies with the original DataFrame and drop the original column
    df = pd.concat([df, theme_dummies], axis=1)

    # --- One-hot encode the "emotion" column ---
    # Create dummy variables for the "emotion" column
    emotion_dummies = pd.get_dummies(df["emotion"], prefix="emotion")
    # Concatenate the dummies with the DataFrame and drop the original column
    df = pd.concat([df, emotion_dummies], axis=1)
    df.drop("emotion", axis=1, inplace=True)

    # --- One-hot encode the "sentiment" column ---
    # Create dummy variables for the "sentiment" column
    sentiment_dummies = pd.get_dummies(df["sentiment"], prefix="sentiment")
    # Concatenate the dummies with the DataFrame and drop the original column
    df = pd.concat([df, sentiment_dummies], axis=1)

    # --- Cyclical encode the "hour_24h" column ---
    # For cyclical encoding, convert the hour into sine and cosine components.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_24h"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_24h"] / 24)

    # --- Renaming new columns to be lowercased and replace spaces with underscores ---
    rename_mapping = dict()
    for col in df.columns:
        if (
            "theme" in col.lower()
            or "emotion" in col.lower()
            or "time_of_day" in col.lower()
        ):
            rename_mapping[col] = col.lower().replace(" ", "_")
    df.rename(columns=rename_mapping, inplace=True)

    # convert boolean values to integers
    # i.e. True -> 1, False -> 0
    for col in df.columns:
        if col in BOOLEAN_COLUMNS:
            df[col] = df[col].astype(int)
            df[col] = df[col].astype(int)
            df[col] = df[col].astype(int)

    # remove rows where "Words_Found" is 0
    before = len(df)
    df = df[df["Words_Found"] != 0]
    print(f"Number of rows with Words_Found = 0: {before - len(df)}")

    return df
