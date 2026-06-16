import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src import config


def load_data():
    return pd.read_csv(config.CLEANED_DATA_FILE)


def get_symptom_columns(df):
    return [col for col in df.columns if col != config.TARGET_COL]


def add_symptom_count(df):
    symptom_cols = get_symptom_columns(df)

    df["symptom_count"] = df[symptom_cols].sum(axis=1)

    return df


def encode_target(df):
    encoder = LabelEncoder()

    df["disease_encoded"] = encoder.fit_transform(
        df[config.TARGET_COL]
    )

    joblib.dump(
        encoder,
        config.LABEL_ENCODER_FILE
    )

    return df, encoder


def save_dataset(df):
    df.to_csv(
        config.CLEANED_DATA_FILE,
        index=False
    )


def run_feature_engineering():

    df = load_data()

    df = add_symptom_count(df)

    df, encoder = encode_target(df)

    save_dataset(df)

    print(
        f"Feature engineering completed."
    )

    print(
        f"Total diseases: {len(encoder.classes_)}"
    )

    print(
        f"Dataset shape: {df.shape}"
    )


if __name__ == "__main__":
    run_feature_engineering()