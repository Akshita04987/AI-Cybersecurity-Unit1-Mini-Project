"""Task 2 - Exploratory Data Analysis of the NSL-KDD dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, FIG_DIR, FEATURES, CAT_COLS

sns.set_theme(style="whitegrid")
PAL = {"Normal": "#4C9F70", "Suspicious": "#D64045"}
df = load_data()
df["class"] = df["label"].map({0: "Normal", 1: "Suspicious"})

# --- Overview -----------------------------------------------------------------
print("Shape:", df.shape)
print("Missing values (total):", int(df.isna().sum().sum()))
print("\nClass balance:\n", df["class"].value_counts(normalize=True).round(3))
print("\nAttack categories:\n", df["attack_type"].value_counts())
num_feats = [f for f in FEATURES if f not in CAT_COLS]
print("\nCorrelation of each numeric feature with 'suspicious':\n",
      df[num_feats + ["label"]].corr()["label"].drop("label").abs().sort_values(ascending=False).round(3))

# --- Plot 1: Normal vs suspicious ------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="class", hue="class", palette=PAL, legend=False)
plt.title("Normal vs Suspicious Connections")
plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: Frequency of attack categories --------------------------------------
plt.figure(figsize=(7, 4))
order = df["attack_type"].value_counts().index
sns.countplot(data=df, x="attack_type", order=order, hue="attack_type", legend=False)
plt.yscale("log"); plt.title("Frequency of Connection Types (log scale)")
plt.savefig(FIG_DIR / "02_attack_categories.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 3: Distribution of attack-related features ------------------------------
feats = ["src_bytes", "count", "serror_rate", "same_srv_rate"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, f in zip(axes.ravel(), feats):
    sns.histplot(data=df, x=f, hue="class", bins=40, log_scale=(f == "src_bytes"),
                 element="step", stat="density", common_norm=False, ax=ax, palette=PAL)
    ax.set_title(f"Distribution of {f}")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_feature_distributions.png", dpi=150)
plt.close()

# --- Plot 4: Suspicious rate by protocol (codes 1/2/3) -----------------------------
rate = df.groupby("protocol_type")["label"].mean()
plt.figure(figsize=(5, 4))
rate.plot(kind="bar", color="#D64045", rot=0)
plt.ylabel("Suspicious rate"); plt.xlabel("protocol_type (code)")
plt.title("Suspicious Rate by Protocol Type")
plt.savefig(FIG_DIR / "04_protocol_rate.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 5: Correlation heatmap of key features -----------------------------------
key = num_feats + ["label"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[key].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (key features)")
plt.savefig(FIG_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nEDA plots saved to figures/")
