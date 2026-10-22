# AI-Driven Cyber Threat Awareness and Detection Prototype
**Course:** AI in Cyber Security Lab (ENSP355) | **Assignment:** Lab 01 (Unit 1)

A mini project that explores a cyber-security events dataset, classifies events as
*normal* or *suspicious* with classical ML, and compares the result with a neural network.

## Project structure
```
data/        KDDTrain+.csv, KDDTest+.csv (NSL-KDD, raw records)
src/         00_validate_setup.py, common.py,
             task2_eda.py, task3_ml.py, task4_dl.py
figures/     exported plots (EDA, confusion matrices, ML vs DL)
results/     metrics (JSON / CSV)
```

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/00_validate_setup.py   # should print "Setup OK"
```

## Usage
```bash
python src/task2_eda.py           # Task 2: EDA + plots
python src/task3_ml.py            # Task 3: Logistic Regression, Decision Tree & Random Forest
python src/task4_dl.py            # Task 4: MLPClassifier + ML vs DL comparison
```

## Dataset
**NSL-KDD** (2009) - an improved version of KDD Cup 1999 with duplicate records removed.
- `KDDTrain+` : 125,973 connection records (training)
- `KDDTest+`  : 22,543 connection records (testing; includes attack types not seen in training)
- Original data has 41 features; this project uses a **simplified set of 12**: duration, protocol_type, flag,
  src_bytes, dst_bytes, num_failed_logins, logged_in, count, srv_count, serror_rate, same_srv_rate, dst_host_count
- Attack categories: DoS, Probe, R2L, U2R, plus Normal. For this project they are merged into a
  binary label: **normal (0)** vs **suspicious (1)**.

CSV copy source: https://github.com/InitRoot/NSLKDD-Dataset (Botes & Leenen), in which
`protocol_type` and `flag` are integer-coded; the code treats them as categorical.

**References**
- M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," IEEE CISDA, 2009.
- https://www.unb.ca/cic/datasets/nsl.html

## Pipeline
Median imputation + StandardScaler (numeric) -> most-frequent imputation + OneHotEncoder (categorical)
-> model. Official train/test split (train on KDDTrain+, test on KDDTest+), random_state=42.
