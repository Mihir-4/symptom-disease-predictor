"""
config.py
Central configuration file for the Symptom-to-Disease Predictor project.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

APP_DIR = PROJECT_ROOT / "app"

# ============================================================
# DATA FILES
# ============================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "cleaned_dataset.csv"
)

# ============================================================
# COLUMN CONFIGURATION
# ============================================================

TARGET_COL = "diseases"

# ============================================================
# RANDOMNESS
# ============================================================

RANDOM_STATE = 42

# ============================================================
# DATA SPLITS
# ============================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# ============================================================
# FILTERING
# ============================================================

MIN_SAMPLES_PER_DISEASE = 10

# ============================================================
# CONFIDENCE LEVELS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 70
MEDIUM_CONFIDENCE_THRESHOLD = 40

# ============================================================
# MODEL FILES
# ============================================================

DECISION_TREE_MODEL = (
    MODELS_DIR / "decision_tree.pkl"
)

RANDOM_FOREST_MODEL = (
    MODELS_DIR / "random_forest.pkl"
)

XGBOOST_MODEL = (
    MODELS_DIR / "xgboost.pkl"
)

LIGHTGBM_MODEL = (
    MODELS_DIR / "lightgbm.pkl"
)

BEST_MODEL = (
    MODELS_DIR / "best_model.pkl"
)

LABEL_ENCODER_FILE = (
    MODELS_DIR / "label_encoder.pkl"
)

# ============================================================
# REPORT FILES
# ============================================================

DATA_CLEANING_REPORT = (
    REPORTS_DIR / "data_cleaning_report.md"
)

EVALUATION_REPORT = (
    REPORTS_DIR / "evaluation_report.md"
)

SHAP_REPORT = (
    REPORTS_DIR / "shap_report.md"
)

# ============================================================
# FIGURES
# ============================================================

DISEASE_DISTRIBUTION_FIG = (
    FIGURES_DIR / "disease_distribution.png"
)

SYMPTOM_FREQUENCY_FIG = (
    FIGURES_DIR / "symptom_frequency.png"
)

SYMPTOM_COUNT_FIG = (
    FIGURES_DIR / "symptom_count_distribution.png"
)

CLASS_IMBALANCE_FIG = (
    FIGURES_DIR / "class_distribution_logscale.png"
)

CONFUSION_MATRIX_FIG = (
    FIGURES_DIR / "confusion_matrix.png"
)

CALIBRATION_CURVE_FIG = (
    FIGURES_DIR / "calibration_curve.png"
)

SHAP_SUMMARY_FIG = (
    FIGURES_DIR / "shap_summary.png"
)

FEATURE_IMPORTANCE_FIG = (
    FIGURES_DIR / "feature_importance.png"
)

# ============================================================
# STREAMLIT
# ============================================================

APP_TITLE = "Symptom-to-Disease Predictor"

# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_directories():
    """
    Create required project folders if missing.
    """

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        APP_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully.")