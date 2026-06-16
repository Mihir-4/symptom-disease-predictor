import joblib
import pandas as pd
import numpy as np
import re


def load_model():
    return joblib.load("models/best_model.pkl")


def load_encoder():
    return joblib.load("models/label_encoder.pkl")


def clean_feature_name(name):

    name = str(name)

    name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        name
    )

    name = re.sub(
        r"_+",
        "_",
        name
    )

    return name


def build_input_vector(
    selected_symptoms,
    feature_columns
):

    selected = {
        clean_feature_name(x)
        for x in selected_symptoms
    }

    row = {}

    for feature in feature_columns:

        if feature == "symptom_count":
            continue

        row[feature] = (
            1 if feature in selected
            else 0
        )

    row["symptom_count"] = len(
        selected_symptoms
    )

    return pd.DataFrame([row])


def confidence_level(prob):

    if prob >= 0.80:
        return "High"

    elif prob >= 0.50:
        return "Medium"

    return "Low"


def predict_top3(
    model,
    encoder,
    selected_symptoms,
    feature_columns
):

    X = build_input_vector(
        selected_symptoms,
        feature_columns
    )

    probs = model.predict_proba(X)[0]

    idx = np.argsort(
        probs
    )[-3:][::-1]

    results = []

    for i in idx:

        disease = encoder.inverse_transform(
            [i]
        )[0]

        results.append({

            "disease": disease,

            "probability":
                float(probs[i]),

            "confidence":
                confidence_level(
                    float(probs[i])
                )
        })

    return results