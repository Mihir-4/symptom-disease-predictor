import joblib

features = joblib.load(
    "models/feature_columns.pkl"
)

ALL_SYMPTOMS = sorted(
    [
        x.replace("_", " ")
        for x in features
        if x != "symptom_count"
    ]
)