"""Task 4 - Neural network (MLPClassifier) vs classical ML on the same data."""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from common import get_split, build_preprocessor, FIG_DIR, RES_DIR, RANDOM_STATE

X_train, X_test, y_train, y_test = get_split()  # train: KDDTrain+, test: KDDTest+

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                          max_iter=300, early_stopping=True,
                                          random_state=RANDOM_STATE),
}

rows, pipes = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(X_test)
    infer_time = time.perf_counter() - t0
    pipes[name] = pipe

    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "Train time (s)": train_time,
                 "Inference time (s)": infer_time})

table = pd.DataFrame(rows).set_index("Model").round(4)
print(table.to_string())
table.to_csv(RES_DIR / "ml_vs_dl_comparison.csv")

# Plot performance and training time side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
table[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=axes[0], rot=15)
axes[0].set_title("Detection Performance"); axes[0].set_ylim(0.5, 1.0)
table["Train time (s)"].plot(kind="bar", ax=axes[1], color="#E08A1E", rot=15)
axes[1].set_title("Training Time (s)")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_ml_vs_dl.png", dpi=150)
plt.close()

# Loss curve of the MLP
mlp = pipes["Neural Network (MLP)"].named_steps["clf"]
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("MLP Training Loss Curve")
plt.savefig(FIG_DIR / "08_mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison table and plots.")
