"""Shared helpers: data loading and preprocessing pipeline (Tasks 2-4).

Dataset: NSL-KDD (Tavallaee et al., 2009), official KDDTrain+ / KDDTest+ files.
The CSV copy used here is from github.com/InitRoot/NSLKDD-Dataset (Botes & Leenen),
in which protocol_type / flag are integer-coded and the attack type is in `xAttack`.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "KDDTrain+.csv"
TEST_PATH = ROOT / "data" / "KDDTest+.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
RANDOM_STATE = 42

ATTACK_NAMES = {1: "DoS", 2: "U2R", 3: "R2L", 4: "Probe", 5: "Normal"}
# Simplified version: only 12 easy-to-explain features out of the original 41
FEATURES = [
    "duration", "protocol_type", "flag", "src_bytes", "dst_bytes",
    "num_failed_logins", "logged_in", "count", "srv_count",
    "serror_rate", "same_srv_rate", "dst_host_count",
]
# These columns are category codes, NOT quantities -> must be one-hot encoded
CAT_COLS = ["protocol_type", "flag"]


def load_data(path=TRAIN_PATH):
    """Load a raw NSL-KDD CSV and add binary label (0 normal / 1 suspicious)."""
    df = pd.read_csv(path)
    df["attack_type"] = df["xAttack"].map(ATTACK_NAMES)
    df["label"] = (df["xAttack"] != 5).astype(int)
    df[CAT_COLS] = df[CAT_COLS].astype(str)  # treat codes as categories
    return df


def build_preprocessor(X):
    """Impute missing values, one-hot encode categoricals, scale numerics."""
    num_cols = [c for c in X.columns if c not in CAT_COLS]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),                          # normalisation
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),   # encoding
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, CAT_COLS)])


def get_split():
    """Official NSL-KDD split: train on KDDTrain+, evaluate on KDDTest+."""
    tr, te = load_data(TRAIN_PATH), load_data(TEST_PATH)
    return tr[FEATURES], te[FEATURES], tr["label"], te["label"]
