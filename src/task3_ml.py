"""Task 3 - Classical ML threat classification (Logistic Regression, Decision Tree, Random Forest)."""
import json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from common import get_split, build_preprocessor, FIG_DIR, RES_DIR, RANDOM_STATE

X_train, X_test, y_train, y_test = get_split()  # train: KDDTrain+, test: KDDTest+

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = pipe.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "train_time_sec": train_time,
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["Normal", "Suspicious"]))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Normal", "Suspicious"],
                                            cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(FIG_DIR / f"06_cm_{name.replace(' ', '_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()

with open(RES_DIR / "ml_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSummary:", json.dumps(results, indent=2))
